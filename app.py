import os
import re
import json
import time
from datetime import datetime, timedelta
import requests
from flask import Flask, request, render_template
from markupsafe import Markup
from openai import OpenAI

# ---------- Configuration (use env vars in production) ----------
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "REPLACE_WITH_YOUR_SERPAPI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "REPLACE_WITH_YOUR_OPENAI_KEY")

DEFAULT_PAGES = int(os.getenv("PAGES", "2"))   # server-side pagination: pages (1–5)
DEFAULT_NUM   = int(os.getenv("NUM", "40"))    # results per page (1–100)

app = Flask(__name__)

# ---------- Helpers: time filter & sanitization ----------
def within_past_week(date_str: str) -> bool:
    if not date_str:
        return False
    s = date_str.strip().lower()
    if "yesterday" in s:
        return True
    if re.match(r"\d+\s*(minute|min|minutes|mins)\s*ago", s):
        return True
    if re.match(r"\d+\s*(hour|hr|hours|hrs)\s*ago", s):
        return True
    m = re.match(r"(\d+)\s*(day|days)\s*ago", s)
    if m:
        return int(m.group(1)) <= 7
    m = re.match(r"(\d+)\s*(week|weeks)\s*ago", s)
    if m:
        return int(m.group(1)) <= 1

    # Absolute dates like "Oct 4, 2025", "October 4, 2025", "Oct. 4, 2025"
    from datetime import datetime as dt
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%b. %d, %Y"):
        try:
            when = dt.strptime(date_str.strip(), fmt)
            return (dt.now() - when).days <= 7
        except ValueError:
            pass
    return False

def basic_sanitize_ul(html_fragment: str) -> str:
    """
    Very light allowlist: keep only <ul>, <li>, <a>, <strong>, <em>, <code>, <br>.
    Strip scripts & event handlers; block 'javascript:' URLs.
    Also ensure <a> has target="_blank" rel="noopener".
    """
    if not html_fragment:
        return "<ul></ul>"
    s = html_fragment

    # remove scripts and inline handlers
    s = re.sub(r"(?is)<\s*script.*?>.*?<\s*/\s*script\s*>", "", s)
    s = re.sub(r"(?is)\s+on\w+\s*=\s*(['\"]).*?\1", "", s)

    # block javascript: URLs
    s = re.sub(r'(?i)href\s*=\s*([\'"])\s*javascript:[^\'"]*\1', 'href="#"', s)

    # allowlist tags
    allowed = {"ul", "li", "a", "strong", "em", "code", "br"}
    def keep_or_strip(m):
        tag = m.group(1).lower()
        return m.group(0) if tag in allowed else ""
    s = re.sub(r"(?is)</?([a-z0-9]+)(?:\s[^>]*)?>", keep_or_strip, s)

    # add target/rel to links
    def fix_link(m):
        tag = m.group(0)
        # ensure target and rel
        if 'target=' not in tag:
            tag = tag[:-1] + ' target="_blank"' + ">"
        if 'rel=' not in tag:
            tag = tag[:-1] + ' rel="noopener"' + ">"
        return tag
    s = re.sub(r"(?i)<a\b([^>]*)>", lambda m: fix_link(m), s)
    return s

# ---------- Data fetch (SerpAPI Google Search – tbm=nws) ----------
def fetch_and_clean_news(keyword: str, pages: int, num: int):
    """
    Use SerpAPI Google Search (News tab) with pagination, filter to past week.
    Return list of {title, snippet, url, source, date}.
    """
    base_url = "https://serpapi.com/search"
    all_items, seen = [], set()

    for p in range(pages):
        params = {
            "engine": "google",
            "tbm": "nws",
            "q": keyword,
            "hl": "en",
            "gl": "us",
            "num": num,           # <= 100
            "start": p * num,     # pagination
            "api_key": SERPAPI_KEY,
        }
        r = requests.get(base_url, params=params, timeout=25)
        r.raise_for_status()
        data = r.json()

        results = data.get("news_results") or data.get("organic_results") or []
        for it in results:
            title   = (it.get("title") or "").strip()
            snippet = (it.get("snippet") or it.get("description") or "").strip()
            url_    = (it.get("link") or "").strip()
            src     = it.get("source")
            source  = (src.get("name") if isinstance(src, dict) else (src or "")).strip()
            date    = (it.get("date") or "").strip()

            if not within_past_week(date):
                continue
            if url_ and url_ in seen:
                continue
            if url_:
                seen.add(url_)
            if title or snippet or url_:
                all_items.append({
                    "title": title,
                    "snippet": snippet,
                    "url": url_,
                    "source": source,
                    "date": date,
                })
    return all_items

# ---------- LLM formatting (one <ul> fragment) ----------
def llm_ul_fragment(cleaned_items, keyword: str) -> str:
    kw_lower = (keyword or "").lower()
    startup_mode = any(k in kw_lower for k in [
        "startup", "startups", "ai startup", "agentic ai", "founder",
        "seed", "series a", "pre-seed", "pre seed"
    ])

    BASE_RULES = """
Return a STRICTLY VALID MINIMAL HTML FRAGMENT that is ONE <ul>…</ul> ONLY (no <html>, no <head>, no CSS/JS).

Each <li> MUST follow exactly:
<li><strong>Company or Startup Name</strong> — one-sentence summary (neutral, factual). <a href="URL">link</a></li>

General approach (experienced analyst mindset):
- Prioritize recall while keeping basic precision: extract as many materially relevant company/startup mentions as justified.
- Accept fuzzy references when the name isn’t explicit (e.g., “<Founder or Place>’s <domain> startup (uncertain)”)—be concise and avoid speculation beyond title/snippet/source/date.
- If multiple DISTINCT companies are clearly present in one item (partnerships, M&A, lawsuits), you MAY output up to 2 bullets from that item.
- If a company appears in several items, de-duplicate by name and keep the clearest/most recent one.
- When 'snippet' is empty, compose a concise one-sentence summary from title (+ optional source/date) only.
- If an item has no URL, omit the <a> tag entirely (never fabricate links).
- Keep each bullet to ONE sentence; no emojis; no extra commentary.
"""
    STARTUP_EMPHASIS = """
Startup emphasis:
- Elevate AI/tech startups and emerging companies when relevant to the keyword.
- From headlines like “<Famous alum> raises round for new venture”, infer a probable startup bullet even if the brand name is unclear; mark as (uncertain).
"""
    COMPANY_EMPHASIS = """
Company emphasis:
- Include public companies, large tech firms, and notable private companies tied to the user's theme.
- In multi-company stories (e.g., partnerships, acquisitions), you may output up to 2 bullets if both entities are material.
"""

    system_prompt = (
        "You are an experienced, high-recall research analyst (covering VC, public equities, and industry).\n"
        + BASE_RULES
        + (STARTUP_EMPHASIS if startup_mode else "")
        + COMPANY_EMPHASIS
        + "\nOutput constraints: Output ONLY a single <ul>…</ul> block. Nothing before or after it."
    )

    client = OpenAI(api_key=OPENAI_API_KEY)
    user_input = (
        "Task: From the following news items, extract as many distinct company/startup bullets as are materially relevant, "
        "following the rules and format strictly. Use source/date to help write a neutral one-liner if snippet is missing. "
        f"User keyword/theme: {keyword}\n\n"
        + json.dumps(cleaned_items, ensure_ascii=False, indent=2)
    )

    resp = client.responses.create(
        model="gpt-5",
        instructions=system_prompt,
        input=user_input,
    )

    html_fragment = getattr(resp, "output_text", None)
    if not html_fragment:
        chunks = []
        for item in getattr(resp, "output", []) or []:
            if isinstance(item, dict) and item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") == "output_text" and "text" in c:
                        chunks.append(c["text"])
        html_fragment = "\n".join(chunks) if chunks else str(resp)

    frag = (html_fragment or "").strip()
    if "<ul" not in frag.lower():
        lines = [ln.strip("-• \t") for ln in frag.splitlines() if ln.strip()]
        items = "".join(f"<li>{ln}</li>" for ln in lines)
        frag = f"<ul>\n{items}\n</ul>"

    return basic_sanitize_ul(frag)

# ---------- Routes ----------
@app.get("/")
def home():
    # First render with default keyword (AI Startup)
    return render_template("index.html", keyword="AI Startup", ul_fragment=None, meta=None, error=None)

@app.post("/generate")
def generate():
    try:
        keyword = (request.form.get("keyword") or "AI Startup").strip() or "AI Startup"
        pages  = int(request.form.get("pages") or DEFAULT_PAGES)
        num    = int(request.form.get("num") or DEFAULT_NUM)

        t0 = time.time()
        cleaned = fetch_and_clean_news(keyword, pages=pages, num=num)
        ul = llm_ul_fragment(cleaned, keyword)
        t1 = time.time()

        meta = {
            "count": len(cleaned),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": round(t1 - t0, 2),
            "pages": pages,
            "num": num,
            "keyword": keyword,
        }
        return render_template("index.html", keyword=keyword, ul_fragment=Markup(ul), meta=meta, error=None)
    except Exception as e:
        return render_template("index.html", keyword="AI Startup", ul_fragment=None, meta=None, error=str(e))

@app.get("/healthz")
def healthz():
    return {"ok": True}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=True)
