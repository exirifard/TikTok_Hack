#!/usr/bin/env python3
"""
TikTok Collector — batch-friendly CLI with single-session scraping
(scrape → metadata → optional download)

Features
- Reuses ONE Playwright browser session across all identifiers (solve CAPTCHA once)
- Accepts multiple identifiers (#hashtags and/or @users) and/or --id-file
- Per-ID filenames via templates: default {id}_links.csv and {id}_metadata.csv
- Prompt to download (or --download / --no-download)
- Limit downloads per ID with --download-count (e.g., 19)
- Respectful rate behavior with progress bars

Env (.env or system):
  MS_TOKEN, PROXY, COOKIES_FILE, USER_AGENT
"""


from __future__ import annotations

import argparse
import csv
import logging


import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from tqdm.auto import tqdm
from yt_dlp import YoutubeDL
from urllib.parse import quote

import requests
from datetime import datetime, timezone

import sqlite3

import json 
import pandas as pd


from playwright.sync_api import TimeoutError as PlaywrightTimeoutError


import os
from datetime import datetime, timezone



class MetadataLockActive(Exception):
    """Raised when another process is already fetching metadata for this URL."""
    pass



# 12-hour cache TTL
CACHE_TTL_SECONDS = 12 * 3600  # 12 hours, for metadata scraping

def _acquire_json_lock(cache_path: Path, verbose= False) -> Path:
    """
    Create a .lock file next to `cache_path` to signal that this process
    is currently fetching metadata for that JSON.

    If a reasonably fresh lock already exists, raise MetadataLockActive
    so callers can skip hitting TikTok for this URL.
    """
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    now = datetime.now(timezone.utc)

    if lock_path.exists():
        try:
            with lock_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            pid = int(data.get("pid", -1))
            created_at_str = data.get("created_at")
            created_at = datetime.fromisoformat(created_at_str) if created_at_str else None
        except Exception:
            # Corrupt lock → treat as stale and overwrite
            pid = -1
            created_at = None

        age_ok = False
        if created_at:
            age = (now - created_at).total_seconds()
            age_ok = age < LOCK_MAX_AGE_SECONDS

        if verbose: 
            if age_ok:
                # Fresh lock: another process is likely working on this URL
                raise MetadataLockActive(f"Active metadata lock at {lock_path} (pid={pid})")
            else:
                # Stale lock → we will overwrite it
                logger.warning("Stale metadata lock found at %s; overwriting.", lock_path)

    # Create/refresh the lock for this process
    payload = {
        "pid": os.getpid(),
        "created_at": now.isoformat(),
    }
    tmp = lock_path.with_suffix(lock_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(lock_path)

    return lock_path


# ───────────────────────────── Config & Logging ──────────────────────────────
load_dotenv()
ENV_MS_TOKEN     = os.getenv("MS_TOKEN")
ENV_PROXY        = os.getenv("PROXY")
ENV_COOKIES_FILE = os.getenv("COOKIES_FILE")
ENV_USER_AGENT   = os.getenv("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")

from pathlib import Path

# Base directory for SQLite files
SQLITE_DIR = Path("sqlite_dbs")

# Ensure directory exists
SQLITE_DIR.mkdir(parents=True, exist_ok=True)

# Full file paths
SQLIGHT_METADATA = SQLITE_DIR / "metadatas.sqlite"
SQLIGHT_LINK = SQLITE_DIR / "links.sqlite"

DEFAULT_LOG_FILE = "tiktok_tool.log"
LOCK_MAX_AGE_SECONDS = 120  # 1 minutes; adjust if needed
MAX_WORKERS = 4  # for video download
METADATA_WORKERS = 4 # for metadata download


logger = logging.getLogger("tiktok")
logger.setLevel(logging.ERROR)

console = logging.StreamHandler()
console.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)

logger.addHandler(console)



def setup_logging(level: str = "INFO", log_file: Optional[str] = DEFAULT_LOG_FILE) -> None:
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger.handlers.clear()
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

def get_public_ip(timeout: int = 5) -> str:
    """
    Best-effort public IP resolver.
    Returns empty string if it cannot be determined.
    """
    try:
        resp = requests.get("https://api.ipify.org", timeout=timeout)
        if resp.ok:
            return resp.text.strip()
    except Exception:
        pass
    return ""



# ─────────────────────────────── Validators/Regex ────────────────────────────
IDENTIFIER_RE =  re.compile(r"^[\w._-]+$", re.UNICODE) #re.compile(r"^[A-Za-z0-9._]+$")

# old
#VIDEO_URL_RE  = re.compile(r"^https?://www\.tiktok\.com/@[^/]+/video/\d+/?$")
# new
VIDEO_URL_RE  = re.compile(r"^https?://www\.tiktok\.com/@[^/]+/video/\d+/?(?:\?.*)?$")


def parse_identifier(raw: str) -> Tuple[str, str]:
    raw = raw.strip()
    if not raw:
        raise ValueError("Identifier is empty. Use #hashtag, @user, or {keywords}")
    if raw.startswith("#"):
        identifier, id_type = raw[1:], "hashtag"
    elif raw.startswith("@"):
        identifier, id_type = raw[1:], "user"
    elif raw.startswith("{") and raw.endswith("}"):
        identifier, id_type = raw[1:-1].strip(), "keyword"
    else:
        identifier, id_type = raw, "hashtag"  # default
    if not identifier:
        raise ValueError("Identifier cannot be empty")
    return identifier, id_type





import hashlib

def sanitize_id_for_path(identifier: str, id_type: str | None = None) -> str:
    base = re.sub(r"[^\w._-]+", "_", identifier, flags=re.UNICODE)
    if id_type == "keyword":
        # Add a short hash that *does* distinguish case
        h = hashlib.sha1(identifier.encode("utf-8")).hexdigest()[:6]
        return f"{base}__{h}"
    return base


def parse_count(maybe: Optional[str]) -> Optional[int]:
    if maybe is None or str(maybe).strip() == "":
        return None
    try:
        n = int(str(maybe).strip())
    except ValueError:
        raise ValueError("--count must be an integer")
    if n <= 0:
        raise ValueError("--count must be a positive integer")
    return n


# ───────────────────────────── I/O helpers & templates ───────────────────────



import math
import random

def jitter_coordinates(
    lat: float | None,
    lon: float | None,
    radius_m: float,
) -> tuple[float | None, float | None]:
    """
    Add random jitter up to `radius_m` meters around (lat, lon).

    Uses a uniform distribution over the disk (not just radius).
    """
    if lat is None or lon is None or radius_m <= 0:
        return lat, lon

    # Convert meters to km, then to radians over Earth radius
    R_earth_km = 6371.0
    radius_km = radius_m / 1000.0

    # Pick random distance and bearing
    # d in [0, radius_km], weighted so points are uniform inside circle
    d = radius_km * math.sqrt(random.random())
    bearing = 2 * math.pi * random.random()

    lat1 = math.radians(lat)
    lon1 = math.radians(lon)

    # Move d km along the bearing
    lat2 = math.asin(
        math.sin(lat1) * math.cos(d / R_earth_km)
        + math.cos(lat1) * math.sin(d / R_earth_km) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(d / R_earth_km) * math.cos(lat1),
        math.cos(d / R_earth_km) - math.sin(lat1) * math.sin(lat2),
    )

    # Normalize lon to [-180, 180]
    lon2 = (math.degrees(lon2) + 540) % 360 - 180
    lat2 = math.degrees(lat2)

    return lat2, lon2

from geopy.geocoders import Nominatim

def geocode_city(city: str, state: str=None, country: str=None)-> tuple[float | None, float | None]:
    """
    Returns (latitude, longitude) for a given city, state/province, and country.
    If state or country is missing, the function progressively falls back.
    """
    geolocator = Nominatim(user_agent="geo_lookup")

    # Build possible query combinations from most specific → least specific
    queries = []

    if city and state and country:
        queries.append(f"{city}, {state}, {country}")
    if city and state:
        queries.append(f"{city}, {state}")
    if city and country:
        queries.append(f"{city}, {country}")
    if city:
        queries.append(city)

    # Try each query until one succeeds
    for query in queries:
        try:
            location = geolocator.geocode(query)
            if location:
                return (location.latitude, location.longitude)
        except Exception:
            continue

    # If nothing matched
    return None


def ensure_lat_lon_from_args(args: argparse.Namespace) -> None:
    """
    If args.lat/lon are not set but --city/--country are provided,
    call geocode_city and fill args.lat / args.lon in place.
    """
    if getattr(args, "lat", None) is not None and getattr(args, "lon", None) is not None:
        # Explicit lat/lon wins
        return

    city = getattr(args, "city", None)
    state = getattr(args, "state", None) or getattr(args, "province", None)
    country = getattr(args, "country", None)

    if not city:
        return

##
    coords = geocode_city(city, state, country)
    if not coords:
        print(f"⚠️ Could not resolve geolocation for city={city!r}, country={country!r}")
        return

    lat, lon = coords

    jitter_m = getattr(args, "geo_uncertainty_m", 0.0) or 0.0
    if jitter_m > 0:
        j_lat, j_lon = jitter_coordinates(lat, lon, jitter_m)
        print(
            f"📍 Resolved location: {city!r} ({country or 'unknown country'}) "
            f"→ base={lat:.6f},{lon:.6f}, with ±{jitter_m} m jitter → {j_lat:.6f},{j_lon:.6f}"
        )
        lat, lon = j_lat, j_lon
    else:
        print(
            f"📍 Resolved location: {city!r} ({country or 'unknown country'}) "
            f"→ lat={lat:.6f}, lon={lon:.6f}"
        )

    args.lat = lat
    args.lon = lon


import re
from urllib.parse import urlparse, parse_qs, unquote

def extract_tiktok_video_id(url: str) -> str | None:
    """
    Try to extract a TikTok video ID from a URL.

    Handles examples like:
      - https://www.tiktok.com/@user/video/7271234567890123456
      - https://www.tiktok.com/video/7271234567890123456
      - https://www.tiktok.com/@user/video/7271234567890123456?_r=1&...
      - https://m.tiktok.com/v/7271234567890123456.html
      - https://vm.tiktok.com/ZMABCDEFG/ (needs redirect to full URL, but we still
        give a stable ID-like token if possible)
      - URLs with ?video_id=... or ?item_id=...
    """
    # Undo your quote() so we see the real URL
    url = unquote(url)

    parsed = urlparse(url)
    path = parsed.path.rstrip("/")

    # 1) Most common: /@user/video/1234567890 or /video/1234567890
    m = re.search(r"/video/(\d+)", path)
    if m:
        return m.group(1)

    # 2) Some legacy/mobile links: /v/1234567890.html
    m = re.search(r"/v/(\d+)", path)
    if m:
        return m.group(1)

    # 3) Short links like /t/abc123 or /ZMABCDEFG/
    #    (not a numeric ID, but still a stable token if you want to use it)
    m = re.search(r"/t/([A-Za-z0-9]+)", path)
    if m:
        return m.group(1)

    m = re.search(r"/(ZM[0-9A-Za-z]+)", path)  # vm.tiktok.com/ZMABCDEFG/
    if m:
        return m.group(1)

    # 4) Fallback: query parameters sometimes carry an ID
    qs = parse_qs(parsed.query)
    for key in ("video_id", "item_id", "share_item_id"):
        if key in qs and qs[key]:
            return qs[key][0]

    return None


def _is_tiktok_block_error(e: Exception) -> bool:
    """
    Heuristic: detect when TikTok / network is blocking us during metadata fetch.
    You can refine this as you see specific error messages.
    """
    msg = str(e).lower()
    # Common things yt-dlp / network will say when blocked / throttled
    return (
        "http error 403" in msg
        or "forbidden" in msg
        or "429" in msg
        or "too many requests" in msg
        or "ssl: wrong version number" in msg  # some VPN / TLS issues
    )


def _prompt_for_ip_change():
    """
    Ask the user to change IP / VPN and press ENTER to continue.
    """
    print("\n🔴 TikTok likely blocked or throttled metadata requests.")
    print("Please:")
    print("  • Change VPN / IP or wait a bit")
    print("  • Ensure cookies (if used) are still valid")
    input("When you are ready, press ENTER to retry this URL… ")


def append_search_type(filename: str, search_type: str) -> str:
    if not search_type:
        return filename

    root, ext = os.path.splitext(filename)  # keeps ".csv"
    safe = search_type.strip().replace(" ", "_").replace("#", "")
    return f"{root}_{safe}{ext}"


def resolve_filename(template: str, identifier_stem: str, default_template: str) -> str:
    tpl = template or default_template
    if "{id}" in tpl:
        return tpl.format(id=identifier_stem)
    p = Path(tpl)
    stem = p.stem + f"_{identifier_stem}"
    return str(p.with_name(stem + p.suffix))






def save_urls_to_csv(
    urls: Iterable[str],
    filename: str | Path,
    *,
    search_type: Optional[str] = None,
    query: Optional[str] = None,
    scraped_at: Optional[str] = None,
    ip_address: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    logger=logger,
):
    """
    Robust CSV writer with:
      - full try/except protection
      - atomic temp-file writing
      - fallback CSV if primary write fails
      - guaranteed headers
    """


    #append_search_type(filename,search_type)

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    urls_list = list(urls)

    if scraped_at is None:
        scraped_at = datetime.now(timezone.utc).isoformat()


    row_template = (
        search_type or "",
        query or "",
        scraped_at,
        ip_address or "",
        latitude if latitude is not None else "",
        longitude if longitude is not None else "",
    )


    # Data rows
    rows = [[u, *row_template] for u in urls_list]


    # Always write atomically: write to tmp first
    tmp_path = filename.with_suffix(filename.suffix + ".tmp")

    try:
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["url", "search_type", "query", "scraped_at", "ip_address", "latitude","longitude"])
            w.writerows(rows)

        # Atomic replace of original file
        tmp_path.replace(filename)

        logger.info(f"Saved {len(rows)} rows → {filename}")

    except Exception as e:
        logger.error(f"CSV write failed for {filename}: {e}")

        # fallback path
        fallback = filename.with_suffix(".fallback.csv")
        try:
            with open(fallback, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["url", "search_type", "query", "scraped_at", "ip_address","latitude","longitude"])
                w.writerows(rows)

            logger.error(f"Data saved to FALLBACK CSV: {fallback}")

        except Exception as e2:
            logger.critical(
                f"FATAL: Could NOT write main CSV or fallback CSV for {filename}. "
                f"URLs for query='{query}' are LOST. Error: {e2}"
            )




def read_urls_from_csv(filename: str | Path) -> List[str]:
    filename = Path(filename)
    with open(filename, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        urls = [row["url"].strip() for row in r if row.get("url")]
    out: List[str] = []
    seen = set()
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        if VIDEO_URL_RE.match(u):
            out.append(u)
        else:
            logger.warning("Skipping non-TikTok video URL: %s", u)
    return out


def write_links_to_sqlite(
    db_path: str | Path,
    urls: List[str],
    *,
    search_type: str,
    query: str,
    scraped_at: str,
    ip_address: str,
    latitude: float | None = None,
    longitude: float | None = None,
    logger=logger,
    notify_user: bool = False,
) -> None:

    """
    Write scraped link URLs into SQLite.

    NEW BEHAVIOUR:
      - Every scrape is stored as a separate row (no dedupe).
      - No UNIQUE(url, search_type, query) constraint.
      - `row_id` is an autoincrement primary key.
      - Keeps retry + fallback DB logic.
    """
    if not urls:
        return

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    def _insert_into(db_file: str | Path) -> bool:
        """Inner insert logic with its own try block."""
        try:
            conn = sqlite3.connect(str(db_file), timeout=10)
            cur = conn.cursor()

            # Ensure table exists with NON-unique rows, row_id as PK
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS scrape_links (
                    row_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    url         TEXT NOT NULL,
                    search_type TEXT,
                    query       TEXT,
                    scraped_at  TEXT,
                    ip_address  TEXT,
                    latitude    REAL,
                    longitude   REAL
                )

                """
            )

            # Build rows for this scrape
            rows = [
                (u, search_type, query, scraped_at, ip_address, latitude, longitude)
                for u in urls
            ]


            # Plain INSERT: keep *all* scrapes (no OR IGNORE)
            cur.executemany(
                """
                INSERT INTO scrape_links
                    (url, search_type, query, scraped_at, ip_address, latitude, longitude)
                VALUES (?, ?, ?, ?, ?, ?, ?)

                """,
                rows,
            )

            conn.commit()
            conn.close()

            inserted_count = len(rows)

            if notify_user:
                print(f"✔ Added {inserted_count} rows to SQLite → {db_file}")

            logger.info("Inserted %d rows into %s", inserted_count, db_file)
            return True

        except sqlite3.IntegrityError as e:
            # Likely old schema with UNIQUE(url, search_type, query)
            logger.error(
                "IntegrityError in %s: %s. "
                "This usually means the existing 'scrape_links' table still has "
                "a UNIQUE(url, search_type, query) constraint. "
                "To store every scrape, drop that table or delete the DB so it can "
                "be recreated without the UNIQUE constraint.",
                db_file,
                e,
            )
            return False

        except sqlite3.OperationalError as e:
            logger.error("SQLite OperationalError in %s: %s", db_file, e)
            return False

        except Exception as e:
            logger.error("Unexpected DB error in %s: %s", db_file, e)
            return False

    # 1) First attempt
    if _insert_into(db_path):
        return

    logger.warning("First write attempt failed for %s. Retrying…", db_path)

    # 2) Retry same DB
    if _insert_into(db_path):
        logger.info("Recovered: second attempt succeeded for %s", db_path)
        return

    logger.error("Second attempt failed. Saving into fallback DB.")

    # 3) Fallback path
    fallback_path = Path(str(db_path) + ".fallback.sqlite")

    if _insert_into(fallback_path):
        logger.error("Data saved into FALLBACK DB: %s", fallback_path)
        if notify_user:
            print(f"⚠️  Data written to fallback SQLite DB instead → {fallback_path}")
    else:
        logger.critical(
            "FATAL: All attempts to save into SQLite failed. "
            "URLs for query='%s' are LOST unless saved in CSV.",
            query,
        )
        if notify_user:
            print("❌ FATAL: Could not save URLs into SQLite or fallback DB.")

def save_metadata_to_csv(
    results: Iterable[Mapping[str, Any]],
    filename: str | Path,
    *,
    scraped_at: Optional[str] = None,
    logger=None,
) -> None:
    """
    Robust, atomic CSV writer for metadata dicts.

    Features:
      - Creates parent directory if needed
      - Ensures a 'scraped_at' column exists (uses provided or auto-now)
      - Computes a stable set of fieldnames from all rows
      - Atomic write via temp file + replace()
      - Fallback CSV if main write fails (e.g., file locked by Excel)
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    # Normalize to list
    results_list = list(results)

    if not results_list:
        if logger:
            logger.warning("No metadata rows to save → %s", filename)
        return

    if scraped_at is None:
        scraped_at = datetime.now(timezone.utc).isoformat()

    # Ensure every row has scraped_at
    norm_rows: list[dict[str, Any]] = []
    for r in results_list:
        row = dict(r)  # shallow copy
        row.setdefault("scraped_at", scraped_at)
        norm_rows.append(row)

    # Build a stable fieldname list:
    #   - start with keys of first row (preserves "natural" order)
    #   - add any extra keys from other rows, sorted
    key_union: set[str] = set()
    for r in norm_rows:
        key_union.update(r.keys())

    base_keys = list(norm_rows[0].keys())
    extra_keys = sorted(k for k in key_union if k not in base_keys)
    fieldnames = base_keys + extra_keys

    # Atomic temp path
    tmp_path = filename.with_suffix(filename.suffix + ".tmp")

    # ---------- MAIN WRITE ----------
    try:
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(norm_rows)

        # Atomic replace of original file
        tmp_path.replace(filename)

        if logger:
            logger.info(
                "Wrote metadata for %d videos → %s",
                len(norm_rows),
                filename,
            )
        return

    except Exception as e:
        if logger:
            logger.error("CSV write failed for %s: %s", filename, e)

    # ---------- FALLBACK ----------
    fallback = filename.with_suffix(".fallback.csv")

    try:
        with open(fallback, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(norm_rows)

        if logger:
            logger.error("Data saved to FALLBACK CSV → %s", fallback)

    except Exception as e2:
        if logger:
            logger.critical(
                "FATAL: Could NOT write main CSV or fallback CSV for %s. "
                "Metadata LOST. Error: %s",
                filename,
                e2,
            )



def write_metadata_to_sqlite(
    db_path: str | Path,
    records: list[dict[str, Any]],
    logger=logger,
    notify_user: bool = False,
) -> None:
    """
    Write TikTok metadata records into SQLite.

    NEW BEHAVIOUR:
      - Every scrape is stored as a separate row (no REPLACE)
      - 'id' is NOT a primary key anymore; we keep a full history
      - 'row_id' is an autoincrement primary key

    Expects each record to have at least:
      id, title, uploader, artist, uploader_id, upload_date,
      view_count, like_count, repost_count, comment_count,
      description, tags, webpage_url, scraped_at (if missing, filled with now-UTC)
    """
    if not records:
        return

    db_path = str(db_path)

    try:
        conn = sqlite3.connect(db_path, timeout=10)
        cur = conn.cursor()

        # --- Check existing schema (if any) ---
        cur.execute("PRAGMA table_info(tiktok_metadata)")
        info = cur.fetchall()
        cols = [row[1] for row in info]  # row[1] = column name

        if not cols:
            # Table does not exist yet → create NEW schema with row_id as PK
            cur.execute(
                """
                CREATE TABLE tiktok_metadata (
                    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT,
                    title TEXT,
                    uploader TEXT,
                    artist TEXT,
                    uploader_id TEXT,
                    upload_date TEXT,
                    view_count INTEGER,
                    like_count INTEGER,
                    repost_count INTEGER,
                    comment_count INTEGER,
                    description TEXT,
                    tags TEXT,
                    webpage_url TEXT,
                    scraped_at TEXT
                )
                """
            )
            # Optional: index on id for query speed
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_tiktok_metadata_id ON tiktok_metadata(id)"
            )
            cols = [  # match the new schema
                "row_id",
                "id",
                "title",
                "uploader",
                "artist",
                "uploader_id",
                "upload_date",
                "view_count",
                "like_count",
                "repost_count",
                "comment_count",
                "description",
                "tags",
                "webpage_url",
                "scraped_at",
            ]
        else:
            # Table exists; check if it still uses old schema with id as PK
            # PRAGMA table_info: row[5] == 1 means "this column is part of the PRIMARY KEY"
            pk_cols = [row[1] for row in info if row[5] == 1]
            if "id" in pk_cols and "row_id" not in cols:
                # Old schema → you won't get multiple rows per id
                # but we still insert using INSERT (will raise if duplicate id)
                if logger:
                    logger.warning(
                        "tiktok_metadata table uses 'id' as PRIMARY KEY in %s. "
                        "To store multiple rows per video, drop the table or delete the DB "
                        "so it can be recreated with row_id as PK.",
                        db_path,
                    )

            # Ensure scraped_at column exists for older DBs
            if "scraped_at" not in cols:
                cur.execute("ALTER TABLE tiktok_metadata ADD COLUMN scraped_at TEXT")
                cols.append("scraped_at")

        # --- Prepare rows ---
        now_iso = datetime.now(timezone.utc).isoformat

        rows = []
        for r in records:
            scraped_at = r.get("scraped_at") or now_iso()

            rows.append(
                (
                    r.get("id"),
                    r.get("title"),
                    r.get("uploader"),
                    r.get("artist"),
                    r.get("uploader_id"),
                    r.get("upload_date"),
                    r.get("view_count"),
                    r.get("like_count"),
                    r.get("repost_count"),
                    r.get("comment_count"),
                    r.get("description"),
                    r.get("tags"),
                    r.get("webpage_url"),
                    scraped_at,
                )
            )

        # --- Insert WITHOUT REPLACE (every scrape becomes a new row) ---
        cur.executemany(
            """
            INSERT INTO tiktok_metadata (
                id, title, uploader, artist, uploader_id,
                upload_date, view_count, like_count, repost_count,
                comment_count, description, tags, webpage_url, scraped_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

        conn.commit()
        conn.close()

        if notify_user:
            print(f"✔ Metadata: {len(rows)} rows written to SQLite → {db_path}")
        if logger:
            logger.info("Metadata: wrote %d rows into %s", len(rows), db_path)

    except sqlite3.IntegrityError as e:
        # This can happen if you're still on the old schema
        if logger:
            logger.error(
                "IntegrityError writing metadata into %s: %s "
                "(likely because 'id' is still PRIMARY KEY).",
                db_path,
                e,
            )
        if notify_user:
            print(
                f"❌ Metadata: integrity error writing into SQLite → {db_path} "
                f"(see log; you may need to drop tiktok_metadata to allow duplicates)."
            )

    except Exception as e:
        if logger:
            logger.error("Metadata SQLite write failed for %s: %s", db_path, e)
        if notify_user:
            print(f"❌ Metadata: failed to write into SQLite → {db_path}")


def read_identifiers(args: argparse.Namespace) -> List[Tuple[str, str]]:
    """
    Collect identifiers from CLI arguments, optional ID file, and --search keywords.
    Returns a list of tuples: (identifier, id_type)
      id_type ∈ {"hashtag", "user", "keyword"}
    """
    raws: List[str] = []

    # 1) Keyword search phrases from --search
    if getattr(args, "search", None):
        for kw in args.search:
            kw = kw.strip()
            if kw:
                # Wrap in {} to reuse keyword-handling logic
                raws.append(f"{{{kw}}}")

    # 2) Direct identifiers from positional args (#hashtags/@users/{keywords})
    if getattr(args, "identifiers", None):
        raws.extend(args.identifiers)

    # 3) Identifiers read from a file (--id-file)
    if getattr(args, "id_file", None):
        with open(args.id_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                # Allow comment lines starting with "# "
                if s and not s.startswith("# "):
                    raws.append(s)

    # 4) Ensure we have something
    if not raws:
        raise SystemExit(
            "No identifiers provided. Use #hashtags, @users, {keywords}, "
            "or --search 'keyword phrases'."
        )

    # 5) Parse and validate identifiers
    parsed: List[Tuple[str, str]] = []
    for r in raws:
        identifier, id_type = parse_identifier(r)
        parsed.append((identifier, id_type))

    return parsed



# ─────────────────────────── Reusable Playwright Session ─────────────────────
class TikTokScraperSession:
    """
    One Playwright browser+context+page for all identifiers.
    CAPTCHA prompt (if needed) happens at most once per run.
    """
    def __init__(
        self,
        *,
        headless: bool,
        proxy: Optional[str],
        user_agent: Optional[str],
        ms_token: Optional[str],
        pause_for_captcha: bool,
        latitude: float=None,
        longitude: float=None,
        geo_accuracy_m: float = 10.0, 
        device_profile: str = "desktop",
        device: Optional[str] = None,
        browser:  Optional[str] ="chromium",
        persist_cookie_file: Optional[str] = None,
    ):
        self.headless = headless
        self.proxy = proxy

        self.user_agent = user_agent
        self.browser = browser
        self.ms_token = ms_token
        self.pause_for_captcha = pause_for_captcha
        
        self.latitude = latitude
        self.longitude = longitude
        self.geo_accuracy_m = geo_accuracy_m
        
        self.device_profile = device_profile
        self.device = device
        self.persist_cookie_file = persist_cookie_file
        # ✅ Initialize CAPTCHA state
        self._captcha_prompted = False

    def __enter__(self):
        from playwright.sync_api import sync_playwright
        self._p = sync_playwright().start()

        browser_args = {"headless": self.headless}
        if self.proxy:
            browser_args["proxy"] = {"server": self.proxy}

        if self.browser == "chromium":
            self._browser = self._p.chromium.launch(**browser_args)
        elif self.browser == "firefox":
            self._browser = self._p.firefox.launch(**browser_args)
        elif self.browser == "webkit":
            self._browser = self._p.webkit.launch(**browser_args)
        else:
            raise ValueError(f"Unsupported browser: {self.browser}")


        # --- CONTEXT options ---
        ctx_args = {
            "permissions": ["geolocation"],   # allow GPS
        }

        # If user selected a real device profile
        if self.device:
            devices = self._p.devices
            if self.device in devices:
                print(f"📱 Using device profile: {self.device}")
                ctx_args.update(devices[self.device])
            else:
                print(f"⚠️ Unknown device '{self.device}'. Available examples:")
                for d in list(self._p.devices.keys())[:10]:
                    print("   -", d)
                raise ValueError(f"Device '{self.device}' not found in Playwright presets.")

        # --- Device profile tweaks (desktop vs mobile) ---
        if self.device_profile == "mobile":
            # Tell Playwright "I'm a phone" (simplified)
            ctx_args.update({
                    "is_mobile": True,
                    "has_touch": True,
                    "viewport": {"width": 390, "height": 844},  # ~iPhone-ish
                    "device_scale_factor": 3,
                }
            )
        # you can add more profiles later (android, tablet, etc.)

        if self.user_agent:
            ctx_args["user_agent"] = self.user_agent

        # Add geolocation if provided
        if self.latitude is not None and self.longitude is not None:
            ctx_args["geolocation"] = {
                "latitude": float(self.latitude),
                "longitude": float(self.longitude),
                "accuracy": float(self.geo_accuracy_m),
            }
            ctx_args["timezone_id"] = "UTC"   # Optional but helps TikTok geo-behavior
            ctx_args["locale"] = "en-US"      # Optional

        self._ctx = self._browser.new_context(**ctx_args)

        # ---------------------------------------------------------
        # Load cookies from previous run, if the file exists
        # ---------------------------------------------------------
        cookie_file = getattr(self, "persist_cookie_file", None)
        if cookie_file and os.path.exists(cookie_file):
            try:
                with open(cookie_file, "r", encoding="utf-8") as f:
                    cookies = json.load(f)
                # Only keep TikTok cookies
                cookies = [c for c in cookies if "tiktok.com" in c.get("domain", "")]
                if cookies:
                    self._ctx.add_cookies(cookies)
                    print(f"🍪 Loaded {len(cookies)} stored cookies from {cookie_file}")
            except Exception as e:
                print(f"⚠️ Could not load cookies: {e}")


        if self.ms_token:
            self._ctx.add_cookies([{
                "name": "msToken",
                "value": self.ms_token,
                "domain": ".tiktok.com",
                "path": "/"
            }])

        self.page = self._ctx.new_page()
        return self


    def __exit__(self, exc_type, exc, tb):
        # Save cookies before shutting down
        try:
            cookie_file = getattr(self, "persist_cookie_file", None)
            if cookie_file and self._ctx:
                cookies = self._ctx.cookies()
                with open(cookie_file, "w", encoding="utf-8") as f:
                    json.dump(cookies, f, indent=2, ensure_ascii=False)
                print(f"💾 Saved {len(cookies)} TikTok cookies → {cookie_file}")
        except Exception as e:
            print(f"⚠️ Could not save cookies: {e}")
        try:
            if self._browser:
                self._browser.close()
        finally:
            if self._p:
                self._p.stop()

    def _maybe_prompt_captcha_once0(self):
        if not self.pause_for_captcha or self._captcha_prompted:
            return
        try:
            self.page.wait_for_selector('a[href*="video"]', timeout=2000)
        except Exception:
            input("\n If a CAPTCHA appeared, solve it in the browser, then press Enter here…")
        self._captcha_prompted = True

    def _maybe_prompt_captcha_once(self):
        """
        Detects when TikTok stops loading video anchors (a possible CAPTCHA),
        and prompts the user with a high-visibility colored message.
        """
        if not self.pause_for_captcha or self._captcha_prompted:
            return

        try:
            self.page.wait_for_selector('a[href*="video"]', timeout=2000)
        except Exception:
            # ANSI colors
            YELLOW = "\033[93m"
            BLUE = "\033[94m"
            GREEN = "\033[92m"
            RESET = "\033[0m"

            msg = f"""
    {YELLOW}────────────────────────────────────────────────────────────{RESET}
    {YELLOW}⚠️  CAPTCHA Detected! TikTok is asking for human verification.{RESET}
    {YELLOW}────────────────────────────────────────────────────────────{RESET}

    {BLUE}Please switch to the browser window and solve the CAPTCHA.{RESET}

    Once the page finishes loading videos again, press {GREEN}Enter{RESET} to continue… 
    """
            input(msg)

        self._captcha_prompted = True


    def scrape_one(self, identifier: str, id_type: str, *, count: Optional[int],
                   timeout_ms: int = 60000, max_scroll_idle: int = 2) -> List[str]:
        encoded = quote(identifier, safe="._-")

        if id_type == "hashtag":
            start_url = f"https://www.tiktok.com/tag/{encoded}"
        elif id_type == "user":
            start_url = f"https://www.tiktok.com/@{encoded}"
        elif id_type == "keyword":
            start_url = f"https://www.tiktok.com/search?q={encoded}"
        else:
            raise ValueError(f"Unsupported id_type: {id_type}")

        urls: List[str] = []
        seen = set()

        pbar = tqdm(total=count, desc=f"Scraping {id_type}:{identifier}", unit="url") if count \
               else tqdm(desc=f"Scraping {id_type}:{identifier}", unit="url")

        try:
            self.page.goto(start_url, timeout=timeout_ms)
        except PlaywrightTimeoutError as e:
            logger.error(
                "Page.goto timeout for %s (%s) → %s",
                identifier,
                id_type,
                e,
            )
            # Bail out for this identifier; caller sees 0 URLs
            return []

        self.page.wait_for_timeout(3000)

        # ───────────────────────────────────────────────────────────
        # Detect “Couldn't find this hashtag/user” BEFORE CAPTCHA
        # (Robust: read full body text, normalize curly apostrophes)
        # ───────────────────────────────────────────────────────────
        try:
            page_text = self.page.inner_text("body").lower()
        except Exception:
            page_text = ""

        # Normalize curly apostrophes → straight apostrophes
        page_text = page_text.replace("’", "'")
 

        if (
            "couldn't find this hashtag" in page_text
            or "couldn't find this account" in page_text 
            or "this user has not published any videos." in page_text 

        ):
            logger.warning("⏩ %s (%s) not found on TikTok — skipping.", identifier, id_type)
            pbar.close()
            return "__HASHTAG_NOT_FOUND__"

        # No “not found” → check normal CAPTCHA flow
        self._maybe_prompt_captcha_once()

        # No “not found” → check normal CAPTCHA flow
        self._maybe_prompt_captcha_once()



        idle_loops = 0
        while count is None or len(urls) < count:
            prev = len(urls)
            self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            self.page.wait_for_timeout(1500)
            for a in self.page.query_selector_all('a[href*="/video/"]'):
                href = a.get_attribute("href") or ""
                full = href if href.startswith("http") else f"https://www.tiktok.com{href}"
                if VIDEO_URL_RE.match(full) and full not in seen:
                    seen.add(full)
                    urls.append(full)
                    pbar.update(1)
                    if count and len(urls) >= count:
                        break
            if len(urls) == prev:
                idle_loops += 1
            else:
                idle_loops = 0
            if idle_loops >= max_scroll_idle:
                break

        pbar.close()
        return urls[:count] if count else urls


# ───────────────────────────── Metadata & Download ───────────────────────────

import json
from datetime import datetime, UTC

from datetime import datetime, UTC
from urllib.parse import quote  # already imported earlier
from pathlib import Path
import json
from typing import Optional
from yt_dlp import YoutubeDL


def fetch_tiktok_metadata(url: str, cookies_file: Optional[str], user_agent: str,  max_age_hours: float = 12, verbose = False) -> dict:
    """
    Fetch TikTok metadata for a single URL.

    Caching rule:
      - Look for a JSON cache file in ./json/ named by (url, today).
      - If it exists and is from today, load it and DO NOT call TikTok.
      - Otherwise, call yt-dlp, save JSON, and return fresh data.
    """
    json_dir = Path("json")
    json_dir.mkdir(exist_ok=True)

    # One cache file per URL (no timestamp in name)
    safe_url = quote(url, safe="")
    video_id = extract_tiktok_video_id(url)

    if video_id:
        cache_name = f"{video_id}.json"
    else:
        # fallback to full safe_url if no ID found
        cache_name = f"{safe_url}.json"
    cache_path = json_dir / cache_name


    # ───────────────────── Try cache first ─────────────────────

    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)

            info = None
            scraped_at = None
            use_cache = False

            # ─────────────────────────────────────────────
            # NEW STYLE: list of {scraped_at, metadata}
            # ─────────────────────────────────────────────
            if isinstance(payload, list):
                freshest_dt = None
                freshest_entry = None

                for entry in payload:
                    if not isinstance(entry, dict):
                        continue

                    # Be tolerant of typos: "metadata" vs "metadat"
                    entry_metadata = entry.get("metadata") or entry.get("metadat") or entry
                    entry_scraped_at = entry.get("scraped_at") or entry_metadata.get("scraped_at")

                    if not entry_scraped_at:
                        continue

                    try:
                        dt = datetime.fromisoformat(entry_scraped_at.replace("Z", "+00:00"))
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=UTC)
                    except Exception:
                        continue

                    age_hours = (datetime.now(UTC) - dt).total_seconds() / 3600.0
                    if age_hours < max_age_hours:
                        # Candidate: fresh enough
                        if freshest_dt is None or dt > freshest_dt:
                            freshest_dt = dt
                            freshest_entry = (entry_metadata, entry_scraped_at)

                if freshest_entry is not None:
                    info, scraped_at = freshest_entry
                    use_cache = True
                else:
                    logger.info(
                        "All cache entries in %s are older than %s hours → re-fetching.",
                        cache_path, max_age_hours
                    )

            # ─────────────────────────────────────────────
            # OLD STYLE: single dict
            # ─────────────────────────────────────────────
            elif isinstance(payload, dict) and "metadata" in payload:
                info = payload["metadata"]
                scraped_at = payload.get("scraped_at")

            else:
                # Very old style: payload == metadata
                info = payload
                scraped_at = payload.get("scraped_at") if isinstance(payload, dict) else None

            # If we didn't already decide use_cache from the list case,
            # do the usual single-record freshness check
            if not use_cache and scraped_at:
                try:
                    dt = datetime.fromisoformat(scraped_at.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=UTC)
                    age_hours = (datetime.now(UTC) - dt).total_seconds() / 3600.0
                    if age_hours < max_age_hours:
                        use_cache = True
                    else:
                        logger.info(
                            "Cache for %s is older than %s hours → re-fetching.",
                            cache_path, max_age_hours
                        )
                except Exception:
                    use_cache = False

            if use_cache and info is not None:
                # Build your standard lightweight return dict
                return {
                    "id": info.get("id"),
                    "title": info.get("title"),
                    "uploader": info.get("uploader"),
                    "artist": info.get("artist"),
                    "uploader_id": info.get("uploader_id"),
                    "upload_date": info.get("upload_date"),
                    "view_count": info.get("view_count"),
                    "like_count": info.get("like_count"),
                    "repost_count": info.get("repost_count"),
                    "comment_count": info.get("comment_count"),
                    "description": info.get("description"),
                    "tags": ",".join(info.get("tags") or []),
                    "webpage_url": info.get("webpage_url"),
                    "scraped_at": scraped_at,
                    "json_file": str(cache_path),
                    "_from_cache": True,
                }

        except Exception as e:
            logger.warning(
                "Failed to read metadata cache %s: %s; refetching.",
                cache_path, e
            )

# ───────────────────── No valid cache → call TikTok with lock ─────────────────────
    lock_path: Optional[Path] = None
    try:
        # Acquire per-URL lock. If another fresh lock exists, this raises MetadataLockActive.
        lock_path = _acquire_json_lock(cache_path)

        opts = {
            "skip_download": True,
            "quiet": True,
            "nocheckcertificate": True,
            "noprogress": True,
            "http_headers": {"User-Agent": user_agent},
        }
        if cookies_file:
            opts["cookiefile"] = cookies_file

        scraped_at = datetime.now(UTC).isoformat()

        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)

        # Save full metadata + scraped_at to cache
        # Save into {id}.json as a growing list of {"scraped_at": ..., "metadata": {...}}

        video_id = info.get("id")
        if video_id:
            video_json_path = json_dir / f"{video_id}.json"

            # Load existing list
            if video_json_path.exists():
                try:
                    with video_json_path.open("r", encoding="utf-8") as f:
                        existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = []
                except Exception:
                    existing = []
            else:
                existing = []

            # Append new entry
            existing.append({
                "scraped_at": scraped_at,
                "metadata": info
            })

            # Save back
            with video_json_path.open("w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
        # ─────────────────────────────────────────────────────────────────────────────


        # Return the light dict used by your pipeline
        return {
            "id": info.get("id"),
            "title": info.get("title"),
            "uploader": info.get("uploader"),
            "artist": info.get("artist"),
            "uploader_id": info.get("uploader_id"),
            "upload_date": info.get("upload_date"),
            "view_count": info.get("view_count"),
            "like_count": info.get("like_count"),
            "repost_count": info.get("repost_count"),
            "comment_count": info.get("comment_count"),
            "description": info.get("description"),
            "tags": ",".join(info.get("tags") or []),
            "webpage_url": info.get("webpage_url"),
            "scraped_at": scraped_at,
            "json_file": str(cache_path),
            "_from_cache": False,
        }

    finally:
        # Always clean up our lock if we created one
        if lock_path is not None:
            try:
                lock_path.unlink(missing_ok=True)
            except Exception as e:
                if verbose:
                    logger.warning("Failed to remove metadata lock %s: %s", lock_path, e)






def batch_fetch_metadata(urls: Iterable[str], out_csv: str | Path,
                         cookies_file: Optional[str], user_agent: str,
                         db_path: str = SQLIGHT_METADATA, consecutive_errors_before_deciding_being_blocked=4, verbose= False) -> str:
    out_csv = str(out_csv)
    results = []

    # One timestamp for this metadata batch
    scraped_at = datetime.now(timezone.utc).isoformat()

    pbar = tqdm(list(urls), desc=f"Metadata → {Path(out_csv).name}", unit="vid")


    consecutive_block_errors = 0  # outside the loop, before `for url in pbar`


    
    for url in pbar:
        while True:
            try:
                info = fetch_tiktok_metadata(url, cookies_file, user_agent)
                # attach batch-level scraped_at
                info["scraped_at"] = scraped_at
                results.append(info)

                # success → reset streak and go to next URL
                consecutive_block_errors = 0
                break

            except MetadataLockActive as e:
                # Another process is already fetching this URL's metadata.
                # Do NOT hit TikTok; just skip this URL in this run.
                logger.info("Skipping metadata for %s due to active lock: %s", url, e)
                consecutive_block_errors = 0
                break

            except Exception as e:
                if verbose:
                    logger.error("Metadata error for %s: %s", url, e)

                if _is_tiktok_block_error(e):
                    consecutive_block_errors += 1

                    if consecutive_block_errors >= consecutive_errors_before_deciding_being_blocked:
                        _prompt_for_ip_change()
                        consecutive_block_errors = 0

                    # retry same URL after (possibly) changing IP
                    continue
                else:
                    # non-block error → reset streak and give up on this URL
                    consecutive_block_errors = 0
                    break
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)



    if results:
        # Separate rows by origin
        fresh_rows = [dict(r) for r in results if not r.get("_from_cache")]
        cached_rows = [dict(r) for r in results if r.get("_from_cache")]

        # ----------------------------------------------------------
        # 1) CSV write ONLY IF at least one row is fresh (TikTok)
        # ----------------------------------------------------------
        if fresh_rows:
            # CSV should not contain _from_cache
            for row in fresh_rows:
                row.pop("_from_cache", None)
            for row in cached_rows:
                row.pop("_from_cache", None)

            # Write ALL rows (fresh + cached) to CSV
            save_metadata_to_csv(fresh_rows + cached_rows, out_csv, logger=logger)
            logger.info("📄 Wrote metadata CSV: %s (fresh rows: %d)", out_csv, len(fresh_rows))
        else:
            logger.info("⏩ No fresh TikTok metadata fetched → skipping CSV write for %s", out_csv)

        # ----------------------------------------------------------
        # 2) SQLite write ONLY for fresh rows
        # ----------------------------------------------------------
        if db_path and fresh_rows:
            write_metadata_to_sqlite(db_path, fresh_rows)
        else:
            logger.info("⏩ Skipping SQLite write (no fresh rows).")

    else:
        logger.warning("No metadata collected → %s", out_csv)

    return out_csv





def download_tiktok_video(url: str, output_dir: str | Path,
                          cookies_file: Optional[str], user_agent: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "nocheckcertificate": True,
        "noprogress": True,
        "http_headers": {"User-Agent": user_agent},
        "merge_output_format": "mp4",
    }
    if cookies_file:
        opts["cookiefile"] = cookies_file
    with YoutubeDL(opts) as ydl:
        ydl.download([url])


def batch_download(urls: List[str], output_dir: str | Path,
                   cookies_file: Optional[str], user_agent: str,
                   max_workers: int = MAX_WORKERS , skip_existing: bool = True) -> None:
    output_dir = Path(output_dir)
    failed: List[Tuple[str, str]] = []

    def worker(u: str) -> None:
        try:
            video_id = u.rstrip("/").split("/")[-1]
            pattern = f"*{video_id}*.mp4"
            if skip_existing and list(output_dir.glob(pattern)):
                return
            download_tiktok_video(u, output_dir, cookies_file, user_agent)
        except Exception as e:
            failed.append((u, str(e)))

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for u in urls:
            futures.append(ex.submit(worker, u))
        for _ in tqdm(as_completed(futures), total=len(futures),
                      desc=f"Downloading → {output_dir}", unit="vid"):
            pass

    if failed:
        for u, err in failed:
            logger.error("Download failed for %s: %s", u, err)
    else:
        logger.info("All requested videos downloaded successfully.")


# ────────────────────────────────── CLI ──────────────────────────────────────
class SmartFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    """Shows defaults and preserves newlines for long examples/notes."""

TOP_EPILOG = r"""
EXAMPLES
  # End-to-end for two IDs (asks whether to download):
  python tiktok_tool.py all "#canada" "@nasa" --count 80

  # Non-interactive, download exactly 19 per ID:
  python tiktok_tool.py all "#canada" "@g7" --count 120 --download --download-count 19

  # Only scrape (no metadata / no download):
  python tiktok_tool.py scrape "#canada" --count 60

  # Scrape from a file (one ID per line), write per-ID CSVs into lists/:
  python tiktok_tool.py scrape --id-file ids.txt --out-csv "lists/{id}_links.csv"

  # Generate metadata after scraping (reads per-ID link CSVs):
  python tiktok_tool.py metadata --id-file ids.txt --meta-out-csv "lists/{id}_metadata.csv"

  # Download 10 per ID into per-ID subfolders under downloads/:
  python tiktok_tool.py download --id-file ids.txt --download --download-count 10

  # Save all downloaded videos into a single flat directory (no subfolders):
  python tiktok_tool.py download --id-file ids.txt --download --flat --skip-existing

  # Headless + proxy:
  python tiktok_tool.py all "#canada" --headless --proxy "http://127.0.0.1:8080"

  # Windows PowerShell users: prefer double quotes
  python tiktok_tool.py all "#canada" "@nasa" --download

FILE NAMING & TEMPLATES
  • {id} is replaced with the sanitized identifier stem (e.g., #canada → "canada")
  • Defaults:
      links CSV     → {id}_links.csv
      metadata CSV  → {id}_metadata.csv
      downloads dir → downloads/<id>/
  • You can prefix subfolders, e.g. --out-csv "out/{id}_links.csv"
  • To save all videos into a single flat directory (no per-ID subfolders), add --flat
    (typically used with the download or all subcommands).

DOWNLOADING
  • By default, `all` asks whether to download. Override with:
      --download        (download without asking)
      --no-download     (do not download)
  • Limit per ID with --download-count N (e.g., 19)
  • Use --flat to put all downloaded videos into one directory instead of per-ID subfolders.

ENVIRONMENT VARIABLES (.env supported)
  MS_TOKEN       Value for TikTok msToken cookie (reduces CAPTCHA friction)
  PROXY          Browser proxy (e.g., http://127.0.0.1:8080)
  COOKIES_FILE   Netscape cookie file for yt-dlp (if needed)
  USER_AGENT     Overrides UA for both Playwright context and yt-dlp
""".strip()




SCRAPE_EPILOG = r"""
Examples:
  python tiktok_tool.py scrape "#canada" --count 80
  python tiktok_tool.py scrape "#canada" "@nasa" --count 50 --out-csv "lists/{id}_links.csv"
  python tiktok_tool.py scrape --id-file ids.txt --headless
Notes:
  - One browser session is reused for all IDs; CAPTCHA prompt appears at most once.
  - Use --no-captcha-pause for non-interactive runs (CI).
""".strip()


META_EPILOG = r"""
Examples:
  # After scraping (defaults: {id}_links.csv → {id}_metadata.csv)
  python tiktok_tool.py metadata "#canada"
  python tiktok_tool.py metadata --id-file ids.txt --meta-out-csv "lists/{id}_metadata.csv"
Tips:
  - You can run metadata without scraping if you already have a links CSV.
  - The links CSV must contain a column named "url".
""".strip()


DOWNLOAD_EPILOG = r"""
Examples:
  python tiktok_tool.py download "#canada" --download --download-count 19
  python tiktok_tool.py download --id-file ids.txt --out-csv "lists/{id}_links.csv" --skip-existing
Details:
  - Videos are saved under: downloads/<id>/ by default.
  - Use --output-dir to change the root download folder.
  - --skip-existing avoids re-downloading files already present.
""".strip()


ALL_EPILOG = r"""
Examples:
  # Ask to download; if yes, it asks "how many per ID?"
  python tiktok_tool.py all "#canada" --count 80

  # Force download 19 per ID without prompting:
  python tiktok_tool.py all "#canada" "@g7" --count 200 --download --download-count 19

  # All steps for a list file; write outputs into subfolders:
  python tiktok_tool.py all --id-file ids.txt --out-csv "out/{id}_links.csv" --meta-out-csv "out/{id}_metadata.csv" --download
Notes:
  - Single Playwright session for all IDs → solve CAPTCHA once.
  - Per-ID CSVs: {id}_links.csv and {id}_metadata.csv by default.
""".strip()


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("identifiers", nargs="*", help="Mix #hashtags and @users (e.g., #canada @nasa)")
    p.add_argument("--id-file", help="Text file with identifiers, one per line (#... or @...)")
    p.add_argument("--out-csv", default="{id}_links.csv",
                   help="Links CSV template (supports {id})")
    p.add_argument("--meta-out-csv", default="{id}_metadata.csv",
                   help="Metadata CSV template (supports {id})")
    p.add_argument("--count", type=str, default=None,
                   help="Max URLs to scrape per ID (blank or omit = all)")
    p.add_argument("--download-count", type=int, default=None,
                   help="Max videos to download per ID (default: all from CSV)")
    p.add_argument("--download", action="store_true",
                   help="Download without prompting")
    p.add_argument("--no-download", action="store_true",
                   help="Never download (override prompts)")
    p.add_argument("--output-dir", default="downloads",
                   help="Directory to save videos (per-ID subfolders)")
    p.add_argument("--headless", action="store_true",
                   help="Run browser headless for scraping")
    p.add_argument("--no-captcha-pause", action="store_true",
                   help="Do not prompt to solve CAPTCHA (non-interactive)")
    p.add_argument("--max-scroll-idle", type=int, default=2,
                   help="Stop if no new URLs after N scrolls")
    p.add_argument("--timeout-ms", type=int, default=60000,
                   help="Page goto timeout (ms)")
    p.add_argument("--proxy", default=ENV_PROXY,
                   help="Proxy for Playwright (default: $PROXY)")
    p.add_argument("--cookies-file", default=ENV_COOKIES_FILE,
                   help="Netscape cookie file for yt-dlp (default: $COOKIES_FILE)")
    p.add_argument("--user-agent", default=ENV_USER_AGENT,
                   help="HTTP User-Agent (default: $USER_AGENT)")
    p.add_argument("--ms-token", default=ENV_MS_TOKEN,
                   help="msToken cookie value (default: $MS_TOKEN)")
    p.add_argument("--max-workers", type=int, default=4,
                   help="Parallel download workers")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip downloads that already exist")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG","INFO","WARNING","ERROR"], help="Log level")
    p.add_argument("--log-file", default=DEFAULT_LOG_FILE,
                   help="Log file (blank to disable)")
    p.add_argument("--search", nargs="+",
                    help="Search TikTok by keyword(s) instead of #/@ identifiers (e.g., --search 'quantum computing' 'AI ethics')")
    p.add_argument("--downloads-dir", default="downloads",
                   help="Directory to store all downloaded videos (flat)")
    p.add_argument("--links-dir", default="links",
                   help="Directory to store link CSVs")
    p.add_argument("--metadata-dir", default="metadata",
                   help="Directory to store metadata files")
    p.add_argument("--flat", action="store_true",
                   default=True,
                   help="Save all videos in a single flat directory (no subfolders)")
    p.add_argument("--lat", type=float,
                   help="Latitude for GPS spoofing (overrides --city/--country if set)")
    p.add_argument("--lon", type=float,
                   help="Longitude for GPS spoofing (overrides --city/--country if set)")

    p.add_argument("--city",
                   help="City name used to infer GPS location (e.g., 'Toronto')")
    p.add_argument("--state", "--province",
                   dest="state",    help="State or province name" )
    p.add_argument("--country",
                   help="ISO country code used with --city (e.g., 'CA', 'JP')")
    p.add_argument(
        "--device-profile",
        choices=["desktop", "mobile"],
        default="desktop",
        help="Device profile for Playwright (desktop or mobile).",
    )
    p.add_argument(
        "--geo-uncertainty-m",
        type=float,
        default=0.0,
        help="Random jitter radius (in meters) to add to spoofed GPS coordinates.",
    )

    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Specific mobile device profile (e.g. 'iPhone 13', 'Pixel 7', 'Galaxy S22').",
    )

    p.add_argument(
        "--persist-playwright-cookies",
        default="pw_cookies.json",
        help="Path to store and load Playwright browser cookies."
    )

    p.add_argument(
        "--geo-accuracy-m",
        type=float,
        default=10.0,
        help="GPS accuracy in meters (Playwright geolocation 'accuracy' field). Default = 30",
    )

    p.add_argument(
        "--browser",
        choices=["chromium", "firefox", "webkit"],
        default="chromium",
        help="Choose Playwright browser engine."
    )








def scrape_with_recovery(
    sess,
    *,
    identifier: str,
    id_type: str,
    count,
    timeout_ms,
    max_scroll_idle,
    min_ok_urls: int,
    try_again_max: int,
    try_again_wait_sec: int,
    reload_rounds: int,
    final_grace_sec: int,
) -> list[str]:
    """
    Shared 'scrape with recovery' helper used by both cmd_scrape and cmd_all.

    Strategy:
      1) Initial scrape
      2) Click 'Try again' up to try_again_max times
      3) Reload the page reload_rounds times
      4) If still under min_ok_urls, beep + ask human to fix CAPTCHA / VPN
         then retry once more
    """
    import platform
    import sys
    import time

    def beep():
        """Cross-platform beep used when human help is needed."""
        try:
            if platform.system() == "Windows":
                import winsound
                winsound.Beep(1000, 700)
            else:
                sys.stdout.write("\a")
                sys.stdout.flush()
        except Exception:
            # Don't ever crash just because beep failed
            pass

    def try_click_try_again(sess) -> bool:
        """Try to click TikTok's 'Try again' button if present."""
        page = sess.page
        try:
            loc = page.locator("button:has-text('Try again')")
            if loc.count() == 0:
                loc = page.get_by_role("button", name="Try again")
            if loc.count() == 0:
                loc = page.locator(
                    "button.enu1bnj4.css-13et812-5e6d46e3--Button-5e6d46e3--StyledButton.ebef5j00"
                )
            if loc.count() > 0:
                loc.first.click()
                return True
        except Exception:
            pass
        return False

    # ── Step 0: initial scrape ────────────────────────────────────
    urls = sess.scrape_one(
        identifier,
        id_type,
        count=count,
        timeout_ms=timeout_ms,
        max_scroll_idle=max_scroll_idle,
    )
    # ---- NEW: stop retry & reload for nonexistent hashtags ----
    if urls == "__HASHTAG_NOT_FOUND__":
        logger.warning("⏩ Skipping %s: TikTok says this hashtag/account does not exist.", identifier)
        return []
    if len(urls) >= min_ok_urls:
        return urls

    # ── Step 1: Try "Try again" a few times ───────────────────────
    for attempt in range(1, try_again_max + 1):
        if not try_click_try_again(sess):
            break
        logger.info(
            "Clicked 'Try again' (%d/%d). Waiting %ds…",
            attempt,
            try_again_max,
            try_again_wait_sec,
        )
        sess.page.wait_for_timeout(try_again_wait_sec * 1000)
        urls = sess.scrape_one(
            identifier,
            id_type,
            count=count,
            timeout_ms=timeout_ms,
            max_scroll_idle=max_scroll_idle,
        )
        if urls == "__HASHTAG_NOT_FOUND__":
            logger.warning(
                "⏩ Skipping %s during 'Try again': TikTok says this hashtag/account does not exist.",
                identifier,
            )
            return []
        if len(urls) >= min_ok_urls:
            logger.info("✅ Recovered after 'Try again': %d URLs.", len(urls))
            return urls


    # ── Step 2: Reload cycles ─────────────────────────────────────
    for r in range(reload_rounds):
        logger.info("Reloading page (round %d/%d)…", r + 1, reload_rounds)
        try:
            sess.page.reload()
            sess.page.wait_for_timeout(3000)
        except Exception:
            pass
        urls = sess.scrape_one(
            identifier,
            id_type,
            count=count,
            timeout_ms=timeout_ms,
            max_scroll_idle=max_scroll_idle,
        )
        if urls == "__HASHTAG_NOT_FOUND__":
            logger.warning(
                "⏩ Skipping %s during reload: TikTok says this hashtag/account does not exist.",
                identifier,
            )
            return []
        if len(urls) >= min_ok_urls:
            logger.info("✅ Recovered after reload: %d URLs.", len(urls))
            return urls


    # ── Step 3: Human intervention ────────────────────────────────
    beep()
    if len(urls) == 0:
        logger.warning(
            "⚠️  %s returned ZERO links — likely CAPTCHA/throttling.",
            identifier,
        )
    else:
        logger.warning(
            "⚠️  %s returned only %d link(s) (< %d).",
            identifier,
            len(urls),
            min_ok_urls,
        )

    print("\n🔴 TikTok may have triggered a CAPTCHA or blocked requests.")
    print("Solve CAPTCHA / change VPN / wait, then press ENTER to retry…")
    input("Press ENTER to retry this identifier…")
    time.sleep(final_grace_sec)

    urls = sess.scrape_one(
        identifier,
        id_type,
        count=count,
        timeout_ms=timeout_ms,
        max_scroll_idle=max_scroll_idle,
    )
    if urls == "__HASHTAG_NOT_FOUND__":
        logger.warning(
            "⏩ Skipping %s after manual intervention: TikTok says this hashtag/account does not exist.",
            identifier,
        )
        return []
    return urls

def cmd_list_devices(args):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        print("Available Playwright device presets:\n")
        for name in sorted(p.devices.keys()):
            print(" -", name)



def cmd_scrape(args: argparse.Namespace) -> None:
    """
    Scrape TikTok URLs for each identifier (hashtag/user/keyword).

    UX:
      1) Pre-scan all IDs and print a clean table of those already collected (skip list)
      2) Show a single overall tqdm for the remaining IDs
      3) For each remaining ID:
         - scrape, try "Try again" (≤3x), reload (≤2x), then beep+prompt if still < MIN_OK_URLS
         - overwrite links CSV with latest result
    """
    import platform
    import sys
    import time
    from pathlib import Path
    from tqdm.auto import tqdm

    MIN_OK_URLS = 10
    TRY_AGAIN_MAX = 3
    TRY_AGAIN_WAIT_SEC = 10
    RELOAD_ROUNDS = 2
    FINAL_GRACE_SEC = 2

    setup_logging(args.log_level, None if str(args.log_file).strip() == "" else args.log_file)
    count = parse_count(args.count) if args.count is not None else None

    # Derive lat/lon from --city/--country if explicit values not given
    ensure_lat_lon_from_args(args)


    ids = read_identifiers(args)  # list[(identifier, id_type)]
    scrape_ip = get_public_ip()

    # ---------- Pre-scan: decide which to skip vs process ----------
    def count_existing_rows(csv_path: str) -> int:
        p = Path(csv_path)
        if not p.exists():
            return 0
        try:
            with open(p, "r", encoding="utf-8") as f:
                return max(0, sum(1 for _ in f) - 1)  # minus header
        except Exception:
            return 0

    prescan = []
    for identifier, id_type in ids:
        safe_id = sanitize_id_for_path(identifier, id_type)
        search_type = "search" if id_type == "keyword" else id_type
        links_csv = Path(args.links_dir) / f"{safe_id}_{search_type}_links.csv"

        n_rows = count_existing_rows(links_csv)
        prescan.append({
            "identifier": identifier,
            "type": id_type,
            "csv": links_csv,
            "rows": n_rows
        })

    skipped = [x for x in prescan if x["rows"] >= MIN_OK_URLS]
    to_process = [x for x in prescan if x["rows"] < MIN_OK_URLS]

    # Force re-scrape all identifiers if --force-repeat is used
    if getattr(args, "force_repeat", False):
        logger.info("⚠️  --force-repeat enabled: re-scraping ALL identifiers.")
        to_process = prescan   # scrape everything
        skipped = []           # nothing is skipped


    # ---------- Pretty report for already-collected (skip list) ----------
    if skipped:
        # compute column widths
        name_w = min(50, max(len(x["identifier"]) for x in skipped))
        rows_w = max(4, max(len(str(x["rows"])) for x in skipped))
        print("\n================= Already collected (skipping) =================")
        print(f"{'Identifier'.ljust(name_w)}  {'URLs'.rjust(rows_w)}  CSV")
        print("-" * (name_w + rows_w + 6 + 20))
        for x in skipped:
            id_disp = x["identifier"][:name_w].ljust(name_w)
            rows_disp = str(x["rows"]).rjust(rows_w)
            print(f"{id_disp}  {rows_disp}  {x['csv']}")
        print("================================================================\n")

    # ---------- Scrape the remaining with a single overall tqdm ----------
    with TikTokScraperSession(
        headless=args.headless,
        proxy=args.proxy,
        user_agent=args.user_agent,
        ms_token=args.ms_token,
        pause_for_captcha=not args.no_captcha_pause,
        latitude=args.lat,
        longitude=args.lon,
        geo_accuracy_m=args.geo_accuracy_m, 
        device_profile=args.device_profile,
        device=args.device, 
        browser=args.browser,
        persist_cookie_file=args.persist_playwright_cookies
    ) as sess:
        with tqdm(total=len(to_process), desc="Processing remaining identifiers", unit="id") as overall:
            for x in to_process:
                identifier, id_type, links_csv, prev_rows = x["identifier"], x["type"], x["csv"], x["rows"]

                if prev_rows > 0:
                    logger.info("ℹ️  %s has only %d URL(s) (< %d) — will re-scrape.",
                                identifier, prev_rows, MIN_OK_URLS)

                logger.info("▶  Scraping %s (%s)…", identifier, id_type)

                urls = scrape_with_recovery(
                    sess,
                    identifier=identifier,
                    id_type=id_type,
                    count=count,
                    timeout_ms=args.timeout_ms,
                    max_scroll_idle=args.max_scroll_idle,
                    min_ok_urls=MIN_OK_URLS,
                    try_again_max=TRY_AGAIN_MAX,
                    try_again_wait_sec=TRY_AGAIN_WAIT_SEC,
                    reload_rounds=RELOAD_ROUNDS,
                    final_grace_sec=FINAL_GRACE_SEC,
                )

                scraped_at = datetime.now(timezone.utc).isoformat()
                search_type = "search" if id_type == "keyword" else id_type
                
                #links_csv= append_search_type(filename=links_csv, search_type=search_type)




                save_urls_to_csv(
                    urls=urls,
                    filename=links_csv,
                    search_type=search_type,
                    query=identifier,
                    scraped_at=scraped_at,
                    ip_address=scrape_ip,
                    latitude=args.lat,
                    longitude=args.lon,
                )


                write_links_to_sqlite(
                    db_path=SQLIGHT_LINK,
                    urls=urls,
                    search_type=search_type,
                    query=identifier,
                    scraped_at=scraped_at,
                    ip_address=scrape_ip,
                    latitude=args.lat,
                    longitude=args.lon,
                )



                n = len(urls)
                if n >= MIN_OK_URLS:
                    logger.info("✅ %s → %d URLs (OK).", identifier, n)
                else:
                    logger.warning("❗ %s → %d URLs (< %d).", identifier, n, MIN_OK_URLS)

                overall.update(1)




def cmd_metadata(args: argparse.Namespace) -> None:
    setup_logging(args.log_level, None if str(args.log_file).strip() == "" else args.log_file)
    ids = read_identifiers(args)

    Path(args.metadata_dir).mkdir(parents=True, exist_ok=True)
    Path(args.links_dir).mkdir(parents=True, exist_ok=True)

    for identifier, id_type in ids:
        safe_id = sanitize_id_for_path(identifier, id_type)
        search_type = "search" if id_type == "keyword" else id_type
        links_csv = Path(args.links_dir) / f"{safe_id}_{search_type}_links.csv"

        if not links_csv.exists():
            links_csv = resolve_filename(args.out_csv, identifier, "{id}_links.csv")
        urls = read_urls_from_csv(links_csv)
        meta_csv = Path(args.links_dir) / f"{safe_id}_{search_type}_metadata.csv"
        batch_fetch_metadata(urls, meta_csv, args.cookies_file, args.user_agent)


def cmd_download(args: argparse.Namespace) -> None:
    setup_logging(args.log_level, None if str(args.log_file).strip() == "" else args.log_file)
    ids = read_identifiers(args)

    Path(args.downloads_dir).mkdir(parents=True, exist_ok=True)
    Path(args.links_dir).mkdir(parents=True, exist_ok=True)

    for identifier, id_type in ids:
        in_csv = Path(args.links_dir) / f"{sanitize_id_for_path(identifier, id_type)}_links.csv"
        if not in_csv.exists():
            in_csv = resolve_filename(args.out_csv, identifier, "{id}_links.csv")

        urls = read_urls_from_csv(in_csv)
        if args.download_count is not None:
            urls = urls[: args.download_count]

        out_dir = Path(args.downloads_dir) if args.flat else Path(args.downloads_dir) / sanitize_id_for_path(identifier, id_type)

        batch_download(
            urls, out_dir, args.cookies_file, args.user_agent,
            max_workers=args.max_workers, skip_existing=args.skip_existing
        )




def cmd_all(args: argparse.Namespace) -> None:
    """
    Run the full pipeline in three global phases:
      1) Scrape all identifiers
      2) Fetch metadata for all identifiers
      3) Download videos for all identifiers (optional)

    This ensures that when you pass multiple IDs, *all* scraping finishes
    first, then *all* metadata is collected, then *all* downloads happen.
    """
    setup_logging(args.log_level, None if str(args.log_file).strip() == "" else args.log_file)

    count = parse_count(args.count) if args.count is not None else None

    ensure_lat_lon_from_args(args)


    scrape_ip = get_public_ip()
    ids = read_identifiers(args)

    args.skip_existing = True #do not download the vide if locally exists
    # Recovery thresholds (same as cmd_scrape)
    MIN_OK_URLS = 10
    TRY_AGAIN_MAX = 3
    TRY_AGAIN_WAIT_SEC = 10
    RELOAD_ROUNDS = 2
    FINAL_GRACE_SEC = 2

    Path(args.downloads_dir).mkdir(parents=True, exist_ok=True)
    Path(args.links_dir).mkdir(parents=True, exist_ok=True)
    Path(args.metadata_dir).mkdir(parents=True, exist_ok=True)

    # ────────────────── Decide download behavior ──────────────────
    download_flag = args.download
    dl_count = args.download_count
    if not args.download and not args.no_download:
        ans = input("Download videos after metadata? [y/N]: ").strip().lower()
        download_flag = ans in {"y", "yes"}
        if download_flag and dl_count is None:
            n = input("How many per ID to download? (Enter for ALL): ").strip()
            if n:
                try:
                    dl_count = max(1, int(n))
                except ValueError:
                    print("Not a number; will download ALL scraped URLs per ID.")

    # We'll keep all URLs in memory, keyed by safe identifier
    urls_by_id: dict[str, list[str]] = {}

    # ───────────────────── Phase 1: SCRAPE ────────────────────────
    with TikTokScraperSession(
        headless=args.headless,
        proxy=args.proxy,
        user_agent=args.user_agent,
        ms_token=args.ms_token,
        pause_for_captcha=not args.no_captcha_pause,
        latitude=args.lat,
        longitude=args.lon,
        device_profile=args.device_profile,
        geo_accuracy_m=args.geo_accuracy_m,
        browser=args.browser,
    ) as sess:
        for identifier, id_type in ids:
            safe_id = sanitize_id_for_path(identifier, id_type)
            logger.info("▶  Scraping %s (%s)…", identifier, id_type)
            search_type = "search" if id_type == "keyword" else id_type
            csv_path = Path(args.links_dir) / f"{safe_id}_{search_type}_links.csv"


            # Skip logic (unless --force is passed)
            

            if not args.force_repeat and csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    if len(df) >= MIN_OK_URLS:
                        logger.info(
                            "⏩ Skipping %s: existing CSV has %d URLs (use --force to override).",
                            identifier,
                            len(df),
                        )
                        urls_by_id[safe_id] = df["url"].dropna().tolist()
                        continue
                except Exception:
                    pass  # fall through to scrape
                
            urls = scrape_with_recovery(
                sess,
                identifier=identifier,
                id_type=id_type,
                count=count,
                timeout_ms=args.timeout_ms,
                max_scroll_idle=args.max_scroll_idle,
                min_ok_urls=MIN_OK_URLS,
                try_again_max=TRY_AGAIN_MAX,
                try_again_wait_sec=TRY_AGAIN_WAIT_SEC,
                reload_rounds=RELOAD_ROUNDS,
                final_grace_sec=FINAL_GRACE_SEC,
            )

            n = len(urls)
            if n >= MIN_OK_URLS:
                logger.info("✅ %s → %d URLs (OK) in `all`.", identifier, n)
            else:
                logger.warning("❗ %s → %d URLs (< %d) in `all`.", identifier, n, MIN_OK_URLS)


            urls_by_id[safe_id] = urls

            search_type = "search" if id_type == "keyword" else id_type
            links_csv = Path(args.links_dir) / f"{safe_id}_{search_type}_links.csv"
            scraped_at = datetime.now(timezone.utc).isoformat()




            save_urls_to_csv(
                urls=urls,
                filename=links_csv,
                search_type=search_type,
                query=identifier,
                scraped_at=scraped_at,
                ip_address=scrape_ip,
                latitude=args.lat,
                longitude=args.lon,
            )

            write_links_to_sqlite(
                db_path=SQLIGHT_LINK,
                urls=urls,
                search_type=search_type,
                query=identifier,
                scraped_at=scraped_at,
                ip_address=scrape_ip,
                latitude=args.lat,
                longitude=args.lon,
            )




    # ─────────────────── Phase 2: METADATA ────────────────────────
    # ─────────────────── Phase 2: METADATA ────────────────────────
    # You can tune this number (4, 6, 8, …). Higher → faster but more risk of
    # rate-limits / SQLite "database is locked".
    

    def _process_metadata_for_identifier(identifier: str, id_type: str) -> None:
        safe_id = sanitize_id_for_path(identifier, id_type)
        urls = urls_by_id.get(safe_id)
        search_type = "search" if id_type == "keyword" else id_type


        # Fallback: read from CSV if not in memory (shouldn't normally happen)
        if urls is None:
            links_csv = Path(args.links_dir) / f"{safe_id}_{search_type}_links.csv"
            urls = read_urls_from_csv(links_csv)
            urls_by_id[safe_id] = urls

        meta_csv = Path(args.metadata_dir) / f"{safe_id}_{search_type}_metadata.csv"
        logger.info("📄 Metadata for %s → %s", identifier, meta_csv)
        batch_fetch_metadata(
            urls,
            meta_csv,
            args.cookies_file,
            args.user_agent,
        )

    logger.info("📄 Phase 2: fetching metadata for %d identifiers with %d workers…",
                len(ids), METADATA_WORKERS)

    with ThreadPoolExecutor(max_workers=METADATA_WORKERS) as executor:
        # Submit one job per identifier
        future_to_id = {
            executor.submit(_process_metadata_for_identifier, identifier, id_type): identifier
            for identifier, id_type in ids
        }

        # Optional: a simple progress loop; tqdm over as_completed if you like
        for future in as_completed(future_to_id):
            identifier = future_to_id[future]
            try:
                future.result()
            except Exception as e:
                logger.error("❌ Metadata phase failed for %s: %s", identifier, e)



    # ─────────────────── Phase 3: DOWNLOAD ────────────────────────
    if download_flag:
        for identifier, id_type in ids:
            safe_id = sanitize_id_for_path(identifier, id_type)
            urls = urls_by_id.get(safe_id)

            if urls is None:
                links_csv = Path(args.links_dir) / f"{safe_id}_links.csv"
                urls = read_urls_from_csv(links_csv)
                urls_by_id[safe_id] = urls

            if dl_count is not None:
                dl_urls = urls[:dl_count]
            else:
                dl_urls = urls

            out_dir = Path(args.downloads_dir) if args.flat else Path(args.downloads_dir) / safe_id
            logger.info(
                "⬇️  Downloading %d videos for %s into %s",
                len(dl_urls),
                identifier,
                out_dir,
            )
            batch_download(
                dl_urls,
                out_dir,
                args.cookies_file,
                args.user_agent,
                max_workers=args.max_workers,
                skip_existing=args.skip_existing,
            )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Scrape TikTok URLs, fetch metadata, and optionally download videos — per #hashtag/@user, with per-ID CSVs.",
        formatter_class=SmartFormatter,
        epilog=TOP_EPILOG,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_scrape = sub.add_parser(
        "scrape",
        help="Scrape video links and save to {id}_links.csv (or template).",
        formatter_class=SmartFormatter,
        epilog=SCRAPE_EPILOG,
    )

    add_common_args(p_scrape)
    p_scrape.add_argument(
        "--force-repeat",
        "--force",
        action="store_true",
        help="Re-scrape all identifiers even if existing CSV already has enough URLs.",
    )
    p_scrape.set_defaults(func=cmd_scrape)

    p_meta = sub.add_parser(
        "metadata",
        help="Read links CSV(s) and write {id}_metadata.csv (or template).",
        formatter_class=SmartFormatter,
        epilog=META_EPILOG,
    )
    add_common_args(p_meta)
    p_meta.set_defaults(func=cmd_metadata)

    p_dl = sub.add_parser(
        "download",
        help="Download from links CSV(s).",
        formatter_class=SmartFormatter,
        epilog=DOWNLOAD_EPILOG,
    )
    add_common_args(p_dl)
    p_dl.set_defaults(func=cmd_download)

    p_all = sub.add_parser(
        "all",
        help="Scrape → metadata → (optional) download.",
        formatter_class=SmartFormatter,
        epilog=ALL_EPILOG,
    )

    p_all.add_argument(
    "--force-repeat",
    "--force",
    action="store_true",
    help="Re-scrape all identifiers even if existing CSV already has enough URLs.", 
    )
    
    add_common_args(p_all)
    p_all.set_defaults(func=cmd_all)

    # --- Utility: list Playwright device profiles ---
    p_dev = sub.add_parser(
        "list-devices",
        help="Show all available Playwright device profiles for use with --device.",
        formatter_class=SmartFormatter,
    )
    p_dev.set_defaults(func=cmd_list_devices)

    p.add_argument("--version", action="version", version="TikTok Collector 2.0.0")

    return p



def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)