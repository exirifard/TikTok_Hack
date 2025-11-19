# TikTok Collector

Batch-friendly TikTok scraping tool — scrape → metadata → download — all using **one Playwright browser session** to minimize CAPTCHA interruptions.

---

## ✨ Features
- Scrape TikTok URLs from hashtags, users, or keyword searches.
- Reuses a single browser session for all identifiers.
- Complete pipeline: scrape links → fetch metadata → download videos.
- Automatic JSON metadata caching (12-hour TTL).
- CSV + SQLite outputs.
- CAPTCHA-aware recovery system.

---

## 📦 Installation

### 1. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate    # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Playwright browsers
```bash
playwright install
```

---

## 🚀 Quick Start

Run the full pipeline:

```bash
python tiktok_tool.py all "#Canada" "@Canada" "{Canada}"
```

This triggers:
1. Scraping
2. Metadata collection
3. Optional download prompt

---

## 🔍 Identifier Types Supported

| Type | Syntax | Example | Description |
|------|--------|---------|-------------|
| Hashtag | `#tag` | `#canada` | Scrapes hashtag page |
| User | `@user` | `@nasa` | Scrapes user page |
| Keyword Search | `{words}` | `{quantum}` | Scrapes TikTok search results |
| Search Option | `--search "text"` | `--search "G7 Canada"` | Same as `{keyword}` |

Examples:
```bash
python tiktok_tool.py all --search "quantum computing"
python tiktok_tool.py all "#g7" "{Canada politics}"
```

---

## 🛠 Core Commands

### 1. Scrape Only
```bash
python tiktok_tool.py scrape "#canada" --count 100
```

### 2. Metadata Only
```bash
python tiktok_tool.py metadata "#canada"
```

### 3. Download Only
```bash
python tiktok_tool.py download "#canada" --download
```

### 4. Full Pipeline (Recommended)
```bash
python tiktok_tool.py all "#canada" "@g7" --count 200 --download
```

---

## 📁 Output Structure

```
project/
├─ links/
│   ├─ canada_hashtag_links.csv
│   └─ nasa_user_links.csv
├─ metadata/
│   ├─ canada_hashtag_metadata.csv
│   └─ nasa_user_metadata.csv
├─ downloads/
│   ├─ canada/
│   └─ nasa/
├─ sqlite_dbs/
│   ├─ links.sqlite
│   └─ metadatas.sqlite
└─ json/
    ├─ 1234567890.json
```

---

## ⚙️ Environment Variables (Optional)

Create a `.env` file:

```
MS_TOKEN=your_token
PROXY=http://127.0.0.1:8080
COOKIES_FILE=cookies.txt
USER_AGENT="Mozilla/5.0 ..."
```

These reduce CAPTCHA and improve extraction reliability.

---

## 🧠 Tips for Better Stability
- Provide a valid `MS_TOKEN`.
- Use a residential proxy or VPN.
- Run without `--headless` for fewer CAPTCHAs.
- Avoid scraping >1000 URLs per run.
- Maintain consistent user-agent.

---

## 📘 Examples

### Scraping mixed identifiers
```bash
python tiktok_tool.py all "#canada" "@unitednations" "{G7 summit}"
```

### Force-download exactly 19 videos per identifier
```bash
python tiktok_tool.py all "#canada" --download --download-count 19
```

### Save downloads in a single flat folder
```bash
python tiktok_tool.py all "#canada" --flat --download
```

### Scrape identifiers from a file
```
# ids.txt
#canada
@NASA
{quantum technology}
```

```bash
python tiktok_tool.py all --id-file ids.txt --download
```

---

## 🏗 Project Metadata

See:
- **pyproject.toml** – project definition
- **setup.py** – packaging & console script configuration
- **tiktok_tool.py** – full CLI implementation

---

## 📄 License
GAC License (as configured in project metadata)
