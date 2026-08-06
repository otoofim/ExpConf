"""
Scrape paper metadata (title, abstract, pdf_link) from CVF Open Access.
Usage:
    python download.py --conf CVPR --year 2024
    python download.py --conf CVPR --year 2024 --max_papers 500
Output: data/{conf}_{year}_papers.json
"""

import argparse
import json
import time
import os
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://openaccess.thecvf.com"

CONF_URLS = {
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

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def get_paper_links(conf: str, year: int) -> list[dict]:
    path = CONF_URLS[conf][year]
    url = BASE_URL + path
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    papers = []
    for tag in soup.select(".ptitle a"):
        href = tag.get("href", "")
        title = tag.get_text(strip=True)
        if href and title:
            papers.append({"title": title, "href": href})
    return papers


def get_paper_details(href: str) -> dict:
    url = BASE_URL + href if href.startswith("/") else href
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    abstract = ""
    abs_div = soup.find("div", id="abstract")
    if abs_div:
        abstract = abs_div.get_text(separator=" ", strip=True)

    pdf_link = ""
    for a in soup.select("a"):
        text = a.get_text(strip=True).lower()
        if "pdf" in text:
            link = a.get("href", "")
            if link.endswith(".pdf"):
                pdf_link = BASE_URL + link if link.startswith("/") else link
                break

    return {"abstract": abstract, "pdf_link": pdf_link}


def scrape(conf: str, year: int, max_papers: int | None = None, delay: float = 0.3) -> list[dict]:
    print(f"Fetching paper list for {conf} {year}...")
    links = get_paper_links(conf, year)
    print(f"  Found {len(links)} papers")

    if max_papers:
        links = links[:max_papers]

    papers = []
    for item in tqdm(links, desc="Scraping abstracts"):
        try:
            details = get_paper_details(item["href"])
            papers.append({
                "title": item["title"],
                "abstract": details["abstract"],
                "pdf_link": details["pdf_link"],
            })
        except Exception as e:
            print(f"  Warning: skipping '{item['title']}': {e}")
        time.sleep(delay)

    return papers


def main():
    parser = argparse.ArgumentParser(description="Scrape CVF Open Access papers")
    parser.add_argument("--conf", default="CVPR", choices=list(CONF_URLS.keys()))
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--max_papers", type=int, default=None,
                        help="Limit number of papers (for testing)")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Seconds between requests")
    args = parser.parse_args()

    if args.year not in CONF_URLS[args.conf]:
        valid = sorted(CONF_URLS[args.conf].keys())
        print(f"Year {args.year} not available for {args.conf}. Valid years: {valid}")
        return

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.conf.lower()}_{args.year}_papers.json"

    if out_path.exists():
        print(f"Output already exists: {out_path}. Delete it to re-scrape.")
        return

    papers = scrape(args.conf, args.year, args.max_papers, args.delay)

    with open(out_path, "w") as f:
        json.dump(papers, f, indent=2)

    print(f"Saved {len(papers)} papers to {out_path}")


if __name__ == "__main__":
    main()
