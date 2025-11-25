# TikTok Collector
**High-Reliability TikTok Scraper with Login, Geo-Spoofing, iPhone Device Emulation & Cookie Persistence**

This tool provides a robust, CAPTCHA-resilient TikTok scraping pipeline designed for research, OSINT, and large-scale data collection. It uses **Playwright**, **mobile device emulation**, **GPS spoofing**, **persistent TikTok login**, **SQLite**, and **JSON caching** — all in one streamlined CLI tool.

---

# ✨ Features

### 🎯 **Scraping Modes**
- Hashtags (`#Canada`)
- User pages (`@canadavibes`)
- Keyword search (`--search "Canada Ottawa"`)
- Video-search (`--search-video`)
- Identifier files (`--id-file ids.txt`)

---

### 🛰 **Advanced Evasion & Stability**
- iPhone / Android device emulation  
- Automatic mobile viewport  
- GPS spoofing with ±meter jitter (`--geo-uncertainty-m 1300`)
- Adjustable location accuracy  
- Human-like scroll timing  
- Reuses a single browser session for all identifiers

---

### 🔐 **Persistent TikTok Login**
- New `login` command  
- Opens TikTok login page inside Playwright  
- Supports: QR code login, password login, Google login  
- Cookies saved to file (`pw_cookies.json`)  
- Automatically reused across sessions  
- Ideal for:
  - Viewing region-locked content  
  - Bypassing TikTok search limitations  
  - Reducing CAPTCHAs

---

### 📦 **Complete Processing Pipeline**
- Scrape → Metadata → Download  
- CSV + JSON + SQLite outputs  
- Automatic metadata caching (12 hours)  
- Progress bars for large datasets  
- Optional flat-folder download

---

### 🛠 CLI Designed for Power Users
- Custom device profiles  
- Browser selection (`--browser chromium|firefox|webkit`)  
- Proxy support  
- Headless/headful modes  
- Crash-tolerant resume  
- Force rescrape (`--force`)

---

# 📦 Installation

Create a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate   # Mac/Linux
```

Install dependencies:

```bash
pip install -r requirements.txt
playwright install
```

---

# 🔐 Login to TikTok (Strongly Recommended)

To avoid CAPTCHAs, restrictions, and missing results, **log in once** and reuse cookies.

### Step 1 — Run login:

```bash
python tiktok_tool.py login \
    --device-profile mobile \
    --device "iPhone 14 Pro" \
    --city "Ottawa" \
    --country "CA" \
    --persist-playwright-cookies "pw_cookies.json"
```

### Step 2 — A browser window opens  
Log in using any method.

### Step 3 — When logged in  
Return to terminal → press **ENTER**  
Cookies are saved and reused automatically.

---

# 📍 Geolocation Spoofing

Specify any city:

```bash
--city "Ottawa"
--state "ON"
--country "CA"
--geo-uncertainty-m 1300
```

This places you within a **1.3 km randomized radius** around Ottawa.

TikTok will believe the request originates inside that radius.

---

# 📱 Mobile Device Emulation

To reduce CAPTCHAs and appear like a real TikTok user:

```bash
--device-profile mobile
--device "iPhone 14 Pro"
```

Emulates:

- iPhone 14 Pro user-agent  
- Touch support  
- Device metrics  
- Mobile network patterns

You can view all available devices:

```bash
python tiktok_tool.py list-devices
```

---

# 🚀 Realistic Ottawa Scraper Example  
*(Highly recommended configuration)*

```powershell
python tiktok_tool.py all \
    --hashtag "Canada" "Ottawa" \
    --search "Canada" "Ottawa" "OTTAWA" \
    --search-video "Canada" "Ottawa" "OTTAWA" \
    --usernames "canadavibes5" \
    --force \
    --city "Ottawa" \
    --state "ON" \
    --country "CA" \
    --geo-uncertainty-m 1300 \
    --device-profile mobile \
    --device "iPhone 14 Pro" \
    --persist-playwright-cookies "pw_cookies.json"
```

---

# 🧠 Identifier Types

| Type | Example | Description |
|------|---------|-------------|
| Hashtag | `#canada` | Scrape TikTok’s hashtag page |
| User | `@nasa` | Scrape a user profile |
| Search | `--search "g7 canada"` | TikTok search results |
| Video Search | `--search-video "quantum"` | Only video results |
| Curly braces | `{quantum computing}` | Shortcut for search |
| File list | `--id-file ids.txt` | Load identifiers from a file |

---

# 🏗 Output Structure

```
project/
│
├─ links/
│   ├─ canada_links.csv
│   ├─ ottawa_links.csv
│
├─ metadata/
│   ├─ canada_metadata.csv
│   └─ ottawa_metadata.csv
│
├─ downloads/
│   ├─ canada/
│   ├─ ottawa/
│
├─ sqlite/
│   ├─ links.sqlite
│   └─ metadata.sqlite
│
└─ json/
    ├─ 755812398127.json
```

---

# ⚙️ Environment Variables (Optional)

Create `.env`:

```
PROXY=http://127.0.0.1:8080
MS_TOKEN=your_ms_token
COOKIES_FILE=pw_cookies.json
USER_AGENT="Mozilla/5.0 ..."
```

---

# 📘 Example PowerShell Script (One-Click)

Create `scrape.ps1`:

```powershell
python tiktok_tool.py all \
    --hashtag "Canada" "Ottawa" \
    --search "Canada" "Ottawa" \
    --search-video "Canada" "Ottawa" \
    --usernames "canadavibes5" \
    --city "Ottawa" \
    --state "ON" \
    --country "CA" \
    --geo-uncertainty-m 1300 \
    --device-profile mobile \
    --device "iPhone 14 Pro" \
    --persist-playwright-cookies "pw_cookies.json" \
    --force
```

Run it with:

```powershell
./scrape.ps1
```

---

# 🧩 Tips for Best Reliability

- Use **mobile mode** for most searches  
- Avoid running in **headless** (TikTok behaves worse)  
- Provide a valid **MS_TOKEN**  
- Enable login & cookie persistence  
- Use residential proxies if scraping intensively  
- Keep `pw_cookies.json` private and out of Git

---

# 📄 License
GAC License (configured in project metadata)

