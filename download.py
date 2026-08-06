"""
Scrape paper metadata (title, abstract, pdf_link) from multiple sources:
  - CVF Open Access  → CVPR, ICCV, ECCV (2018-2022), WACV
  - ECVA             → ECCV 2024+
  - PMLR             → ICML
  - OpenReview       → NeurIPS, ICLR

Usage:
    python download.py --conf CVPR    --year 2026
    python download.py --conf ICML    --year 2025
    python download.py --conf NeurIPS --year 2025
    python download.py --conf ICLR    --year 2026 --max_papers 200
Output: data/{conf}_{year}_papers.json
"""

import argparse
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# ── Source registries ─────────────────────────────────────────────────────────

# CVF Open Access
CVF_BASE = "https://openaccess.thecvf.com"
CVF_PATHS: dict[str, dict[int, str]] = {
    "CVPR": {
        2013: "/CVPR2013.py",
        2014: "/CVPR2014.py",
        2015: "/CVPR2015.py",
        2016: "/CVPR2016.py",
        2017: "/CVPR2017.py",
        2018: "/CVPR2018.py",
        2019: "/CVPR2019.py",
        2020: "/CVPR2020.py?day=all",
        2021: "/CVPR2021.py?day=all",
        2022: "/CVPR2022.py?day=all",
        2023: "/CVPR2023.py?day=all",
        2024: "/CVPR2024.py?day=all",
        2025: "/CVPR2025?day=all",
        2026: "/CVPR2026?day=all",
    },
    "ICCV": {
        2013: "/ICCV2013.py",
        2015: "/ICCV2015.py",
        2017: "/ICCV2017.py",
        2019: "/ICCV2019.py",
        2021: "/ICCV2021.py?day=all",
        2023: "/ICCV2023.py?day=all",
        2025: "/ICCV2025?day=all",
    },
    "ECCV": {
        2018: "/ECCV2018.py",
        2020: "/ECCV2020.py?day=all",
        2022: "/ECCV2022.py?day=all",
    },
    "WACV": {
        2020: "/WACV2020.py",
        2021: "/WACV2021.py",
        2022: "/WACV2022.py",
        2023: "/WACV2023.py",
        2024: "/WACV2024.py",
        2025: "/WACV2025?day=all",
        2026: "/WACV2026?day=all",
    },
}

# ECVA — ECCV 2024+ (no longer mirrored on CVF)
ECVA_BASE = "https://www.ecva.net"
ECVA_CONF_YEARS: dict[str, dict[int, str]] = {
    "ECCV": {
        2024: "eccv_2024",
    },
}

# PMLR — ICML (ICML 2026 volume not yet assigned as of Aug 2026)
PMLR_BASE = "https://proceedings.mlr.press"
PMLR_VOLUMES: dict[str, dict[int, str]] = {
    "ICML": {
        2019: "v97",
        2020: "v119",
        2021: "v139",
        2022: "v162",
        2023: "v202",
        2024: "v235",
        2025: "v267",
    },
}

# OpenReview — NeurIPS and ICLR
# Pre-2025: Blind_Submission covers all submissions (incl. rejected).
# 2025+: Poster covers accepted papers only.
OPENREVIEW_BASE = "https://api.openreview.net"
OPENREVIEW_INVITATIONS: dict[str, dict[int, str]] = {
    "NeurIPS": {
        2021: "NeurIPS.cc/2021/Conference/-/Blind_Submission",
        2022: "NeurIPS.cc/2022/Conference/-/Blind_Submission",
        2023: "NeurIPS.cc/2023/Conference/-/Blind_Submission",
        2024: "NeurIPS.cc/2024/Conference/-/Blind_Submission",
        2025: "NeurIPS.cc/2025/Conference/-/Poster",
    },
    "ICLR": {
        2020: "ICLR.cc/2020/Conference/-/Blind_Submission",
        2021: "ICLR.cc/2021/Conference/-/Blind_Submission",
        2022: "ICLR.cc/2022/Conference/-/Blind_Submission",
        2023: "ICLR.cc/2023/Conference/-/Blind_Submission",
        2024: "ICLR.cc/2024/Conference/-/Blind_Submission",
        2025: "ICLR.cc/2025/Conference/-/Blind_Submission",
        2026: "ICLR.cc/2026/Conference/-/Poster",
    },
}

ALL_CONFS = (
    list(CVF_PATHS.keys())
    + ["ECCV"]          # also has ECVA years
    + list(PMLR_VOLUMES.keys())
    + list(OPENREVIEW_INVITATIONS.keys())
)
# Deduplicate while preserving order
ALL_CONFS = list(dict.fromkeys(ALL_CONFS))


# ── CVF scraper ───────────────────────────────────────────────────────────────

def _cvf_paper_links(conf: str, year: int) -> list[dict]:
    url = CVF_BASE + CVF_PATHS[conf][year]
    soup = BeautifulSoup(
        requests.get(url, headers=HEADERS, timeout=30).text, "html.parser"
    )
    return [
        {"title": tag.get_text(strip=True), "href": tag["href"]}
        for tag in soup.select(".ptitle a")
        if tag.get("href")
    ]


def _cvf_paper_details(href: str) -> dict:
    url = CVF_BASE + href if href.startswith("/") else href
    soup = BeautifulSoup(
        requests.get(url, headers=HEADERS, timeout=30).text, "html.parser"
    )
    abstract = ""
    abs_div = soup.find("div", id="abstract")
    if abs_div:
        abstract = abs_div.get_text(separator=" ", strip=True)

    pdf_link = ""
    for a in soup.select("a"):
        link = a.get("href", "")
        if link.endswith(".pdf"):
            pdf_link = CVF_BASE + link if link.startswith("/") else link
            break

    return {"abstract": abstract, "pdf_link": pdf_link}


def scrape_cvf(conf: str, year: int, max_papers: int | None, delay: float) -> list[dict]:
    print("  Fetching paper list from CVF...")
    links = _cvf_paper_links(conf, year)
    print(f"  Found {len(links)} papers")
    if max_papers:
        links = links[:max_papers]

    papers = []
    for item in tqdm(links, desc="Scraping abstracts"):
        try:
            details = _cvf_paper_details(item["href"])
            papers.append({
                "title": item["title"],
                "abstract": details["abstract"],
                "pdf_link": details["pdf_link"],
            })
        except Exception as e:
            print(f"  Warning: skipping '{item['title']}': {e}")
        time.sleep(delay)
    return papers


# ── ECVA scraper (ECCV 2024+) ─────────────────────────────────────────────────

def scrape_ecva(conf: str, year: int, max_papers: int | None, delay: float) -> list[dict]:
    folder = ECVA_CONF_YEARS[conf][year]
    index_url = f"{ECVA_BASE}/papers.php"
    print(f"  Fetching paper list from ECVA ({index_url})...")
    try:
        soup = BeautifulSoup(
            requests.get(index_url, headers=HEADERS, timeout=30).text, "html.parser"
        )
    except Exception as e:
        print(f"  Error: {e}")
        print("  Note: ecva.net may be blocked from this network.")
        return []

    # Each paper has a title link pointing to its detail page
    paper_links = []
    for a in soup.select("a[href]"):
        href = a["href"]
        if folder in href and href.endswith(".php"):
            paper_links.append({
                "title": a.get_text(strip=True),
                "href": href if href.startswith("http") else ECVA_BASE + "/" + href.lstrip("/"),
            })
    paper_links = list({p["href"]: p for p in paper_links}.values())  # dedup
    print(f"  Found {len(paper_links)} papers")
    if max_papers:
        paper_links = paper_links[:max_papers]

    papers = []
    for item in tqdm(paper_links, desc="Scraping abstracts"):
        try:
            soup = BeautifulSoup(
                requests.get(item["href"], headers=HEADERS, timeout=30).text, "html.parser"
            )
            abstract = ""
            for tag in soup.find_all(["div", "p"]):
                text = tag.get_text(strip=True)
                if len(text) > 100 and tag.get("class") and "abstract" in " ".join(tag.get("class", [])).lower():
                    abstract = text
                    break

            pdf_link = ""
            for a in soup.select("a[href]"):
                if a["href"].endswith(".pdf"):
                    pdf_link = a["href"] if a["href"].startswith("http") else ECVA_BASE + a["href"]
                    break

            papers.append({
                "title": item["title"],
                "abstract": abstract,
                "pdf_link": pdf_link,
            })
        except Exception as e:
            print(f"  Warning: skipping '{item['title']}': {e}")
        time.sleep(delay)
    return papers


# ── PMLR scraper (ICML) ───────────────────────────────────────────────────────

def _pmlr_paper_links(volume: str) -> list[str]:
    url = f"{PMLR_BASE}/{volume}/"
    soup = BeautifulSoup(
        requests.get(url, headers=HEADERS, timeout=30).text, "html.parser"
    )
    return list(dict.fromkeys([
        a["href"] for a in soup.select("a[href]")
        if f"/{volume}/" in a["href"]
        and a["href"].endswith(".html")
        and "github.com" not in a["href"]
    ]))


def _pmlr_paper_details(url: str) -> dict:
    soup = BeautifulSoup(
        requests.get(url, headers=HEADERS, timeout=30).text, "html.parser"
    )
    title_meta = soup.find("meta", {"name": "citation_title"})
    title = title_meta["content"] if title_meta else (soup.title.get_text(strip=True) if soup.title else "")

    abstract = ""
    abs_div = soup.find("div", id="abstract")
    if abs_div:
        text = abs_div.get_text(separator=" ", strip=True)
        m = re.search(r'abstract\s*=\s*\{(.+?)\}\s*(?:}|$)', text, re.DOTALL)
        abstract = m.group(1).strip() if m else text

    pdf_link = ""
    pdf_meta = soup.find("meta", {"name": "citation_pdf_url"})
    if pdf_meta:
        pdf_link = pdf_meta["content"]

    return {"title": title, "abstract": abstract, "pdf_link": pdf_link}


def scrape_pmlr(conf: str, year: int, max_papers: int | None, delay: float) -> list[dict]:
    volume = PMLR_VOLUMES[conf][year]
    print(f"  Fetching paper list from PMLR {volume}...")
    urls = _pmlr_paper_links(volume)
    print(f"  Found {len(urls)} papers")
    if max_papers:
        urls = urls[:max_papers]

    papers = []
    for url in tqdm(urls, desc="Scraping abstracts"):
        try:
            details = _pmlr_paper_details(url)
            papers.append({
                "title": details["title"],
                "abstract": details["abstract"],
                "pdf_link": details["pdf_link"],
            })
        except Exception as e:
            print(f"  Warning: skipping {url}: {e}")
        time.sleep(delay)
    return papers


# ── OpenReview scraper (NeurIPS / ICLR) ──────────────────────────────────────

OPENREVIEW_PAGE_SIZE = 1000


def scrape_openreview(conf: str, year: int, max_papers: int | None, delay: float) -> list[dict]:
    invitation = OPENREVIEW_INVITATIONS[conf][year]
    print(f"  Fetching papers from OpenReview ({invitation})...")

    papers = []
    offset = 0
    while True:
        try:
            resp = requests.get(
                f"{OPENREVIEW_BASE}/notes",
                params={"invitation": invitation, "limit": OPENREVIEW_PAGE_SIZE, "offset": offset},
                headers=HEADERS,
                timeout=30,
            )
            resp.raise_for_status()
        except Exception as e:
            print(f"  Error fetching from OpenReview: {e}")
            print(
                "  Note: openreview.net may be blocked from this network.\n"
                "  Try running download.py from a machine with unrestricted internet access."
            )
            break

        notes = resp.json().get("notes", [])
        if not notes:
            break

        for note in notes:
            content = note.get("content", {})
            title = content.get("title", "")
            abstract = content.get("abstract", "")
            pdf_path = content.get("pdf", "")
            pdf_link = (
                f"https://openreview.net{pdf_path}"
                if pdf_path.startswith("/")
                else pdf_path
            )
            if title:
                papers.append({"title": title, "abstract": abstract, "pdf_link": pdf_link})

        offset += len(notes)
        if max_papers and len(papers) >= max_papers:
            papers = papers[:max_papers]
            break
        if len(notes) < OPENREVIEW_PAGE_SIZE:
            break
        time.sleep(delay)

    print(f"  Retrieved {len(papers)} papers")
    return papers


# ── Dispatch ──────────────────────────────────────────────────────────────────

def valid_years(conf: str) -> list[int]:
    years = set()
    if conf in CVF_PATHS:
        years |= set(CVF_PATHS[conf])
    if conf in ECVA_CONF_YEARS:
        years |= set(ECVA_CONF_YEARS[conf])
    if conf in PMLR_VOLUMES:
        years |= set(PMLR_VOLUMES[conf])
    if conf in OPENREVIEW_INVITATIONS:
        years |= set(OPENREVIEW_INVITATIONS[conf])
    return sorted(years)


def scrape(conf: str, year: int, max_papers: int | None = None, delay: float = 0.3) -> list[dict]:
    # CVF takes priority for ECCV years it hosts (2018-2022)
    if conf in CVF_PATHS and year in CVF_PATHS[conf]:
        return scrape_cvf(conf, year, max_papers, delay)
    if conf in ECVA_CONF_YEARS and year in ECVA_CONF_YEARS[conf]:
        return scrape_ecva(conf, year, max_papers, delay)
    if conf in PMLR_VOLUMES and year in PMLR_VOLUMES[conf]:
        return scrape_pmlr(conf, year, max_papers, delay)
    if conf in OPENREVIEW_INVITATIONS and year in OPENREVIEW_INVITATIONS[conf]:
        return scrape_openreview(conf, year, max_papers, delay)
    raise ValueError(f"No source configured for {conf} {year}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scrape ML conference papers")
    parser.add_argument("--conf", default="CVPR", choices=ALL_CONFS)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--max_papers", type=int, default=None,
                        help="Cap number of papers (useful for testing)")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Seconds between requests")
    args = parser.parse_args()

    years = valid_years(args.conf)
    if args.year not in years:
        print(f"Year {args.year} not available for {args.conf}. Valid: {years}")
        return

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.conf.lower()}_{args.year}_papers.json"

    if out_path.exists():
        print(f"Output already exists: {out_path}. Delete it to re-scrape.")
        return

    print(f"Scraping {args.conf} {args.year}...")
    papers = scrape(args.conf, args.year, args.max_papers, args.delay)

    with open(out_path, "w") as f:
        json.dump(papers, f, indent=2)

    print(f"Saved {len(papers)} papers → {out_path}")


if __name__ == "__main__":
    main()
