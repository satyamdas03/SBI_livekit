# src/ingest_banking.py
import json, time, re
from pathlib import Path
import requests
from bs4 import BeautifulSoup

BANK_DIR = Path("data") / "banking"
SOURCES_JSON = BANK_DIR / "bank_sources.json"

def _sanitize(url: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_.-]", "_", url.split("/")[-1] or "index")
    return name if name.endswith((".pdf", ".txt", ".md")) else f"{name}.txt"

def _fetch(item):
    url = item["url"]
    kind = item.get("kind", "auto")
    fn = item.get("filename") or _sanitize(url)
    out = BANK_DIR / fn

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    ct = resp.headers.get("content-type", "").lower()

    is_pdf = "pdf" in ct or fn.lower().endswith(".pdf")
    if kind == "pdf" or (kind == "auto" and is_pdf):
        out.write_bytes(resp.content)
        return out

    # html/text → .txt
    soup = BeautifulSoup(resp.text, "html.parser")
    text = soup.get_text(separator="\n")
    out = out.with_suffix(".txt")
    out.write_text(text, encoding="utf8")
    return out

def main():
    BANK_DIR.mkdir(parents=True, exist_ok=True)
    items = json.loads(SOURCES_JSON.read_text(encoding="utf8"))
    for it in items:
        try:
            path = _fetch(it)
            print("saved:", path)
            time.sleep(0.4)
        except Exception as e:
            print("FAILED:", it["url"], e)
    print("Done. Start your agent; the FAISS index will be built on first query.")

if __name__ == "__main__":
    main()
