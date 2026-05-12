#!/usr/bin/env python3
"""
Check text / PDF AI score via Yandex Neurodetector.

Usage:
    python3 scripts/check_ai_score.py --text "some text"
    python3 scripts/check_ai_score.py --file path/to/chapter.txt
    python3 scripts/check_ai_score.py --typ path/to/main.typ --section "Введение"
    python3 scripts/check_ai_score.py --pdf path/to/main.pdf
"""

import argparse
import json
import mimetypes
import os
import re
import sys
import urllib.request
import urllib.error
from io import BytesIO

TEXT_API_URL = "https://yandex.ru/lab/neurodetector/api/analyze/text"
FILE_API_URL = "https://yandex.ru/lab/neurodetector/api/analyze/file"
MIN_LEN = 50

# Cookies from browser session (required for auth)
COOKIES = (
    "is_gdpr=0; is_gdpr_b=CIrYeRD/+wIoAg==; yandexuid=2011386511774699380; "
    "yashr=7217822711774699380; "
    "L=ZQpIaXtpX0MEZwYHDH9BQFJOUAcNBENdXzo1Kl0FSxQ9RBQ=.1774852085.1801173.389195.b87d3f3a47e7c6a44f5cf9db905f10e4; "
    "yandex_login=kirill.fdrn; "
    "pi=JpvYDD8s9htl34p8CaKknk00kPqwMpgIBYf2RCUsZ3MfYKGqNmDSp6g/Acr54hEdFLBRKjXThdVFROdJgmnYy01XrqI=; "
    "i=mJD4fHrUOvC1BeRcY+ucmSP5HqnO2UkbkIzDOesKfJedD1DFY2QY08EbrJ+OcjYEb34XLNA+GlDTxx30QjwnXtnYBGg=; "
    "yuidss=2011386511774699380; maps_aadb=1; "
    "Session_id=3:1777631191.5.0.1774852085763:bSOKLg:5ddc.1.2:1|295017209.-1.2.0:3.3:1774852085.6:2088908079.7:1774852085|3:11875238.870513.TO0aL-x0q4rVubZhpxQoa3JmHVU; "
    "sessar=1.1781281.CiBy5nLareogVRgQdyd38ev1YyYa2OUUumlIDZycRNrXQg.pqLxBp3E6BQMKDwVCX3p7XCoNkaQLprDGaSAXwQrFEs; "
    "sessionid2=3:1777631191.5.0.1774852085763:bSOKLg:5ddc.1.2:1|295017209.-1.2.0:3.3:1774852085.6:2088908079.7:1774852085|3:11875238.870513.fakesign0000000000000000000; "
    "yp=2090212085.udn.cDpLaXJhcHJpbnQ%3D#1778344580.szm.1_25:2560x1440:2048x1239; "
    "_yasc=+rIrqZOpZThEImP9QlXYR6Qu7//RgWAzGP24rYN7cWg9qkJgo9qsaaxK5lmEsnxda0SEnQIhHaU=.MTc3ODUyMDYzNDA0NA==; "
    "bh=EjkiQ2hyb21pdW0iO3Y9IjE0OCIsICJCcmF2ZSI7dj0iMTQ4IiwgIk5vdC9BKUJyYW5kIjt2PSI5OSIaBSJ4ODYiKgI/MDICIiI6ByJMaW51eCJCAiIiSgQiNjQiUksiQ2hyb21pdW0iO3Y9IjE0OC4wLjAuMCIsICJCcmF2ZSI7dj0iMTQ4LjAuMC4wIiwgIk5vdC9BKUJyYW5kIjt2PSI5OS4wLjAuMCJaAj8wYLqkiNAGahncyumIDvKst6UL+/rw5w3r//32D9OgzocI"
)

HEADERS = {
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.8",
    "Origin": "https://yandex.ru",
    "Referer": "https://yandex.ru/lab/neurodetector",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
    "Cookie": COOKIES,
}


def build_multipart_boundary():
    import uuid
    return "----" + uuid.uuid4().hex


def check_text(text: str) -> dict:
    text = text.strip()
    if len(text) < MIN_LEN:
        return {"ok": False, "error": f"Text too short ({len(text)} chars, need ≥{MIN_LEN})"}

    payload = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        TEXT_API_URL,
        data=payload,
        headers={**HEADERS, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.read().decode()[:500]}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def check_pdf(pdf_path: str) -> dict:
    """Upload PDF file to Yandex Neurodetector for full-document analysis."""
    if not os.path.isfile(pdf_path):
        return {"ok": False, "error": f"File not found: {pdf_path}"}

    filename = os.path.basename(pdf_path)
    boundary = build_multipart_boundary()

    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    body = BytesIO()
    body.write(f"--{boundary}\r\n".encode())
    body.write(f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode())
    body.write(b"Content-Type: application/pdf\r\n\r\n")
    body.write(file_bytes)
    body.write(f"\r\n--{boundary}--\r\n".encode())
    payload = body.getvalue()

    req = urllib.request.Request(
        FILE_API_URL,
        data=payload,
        headers={
            **HEADERS,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.read().decode()[:500]}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def extract_section_from_typ(typ_path: str, section_title: str) -> str:
    with open(typ_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = rf"^=+\s*{re.escape(section_title)}\s*$"
    match = re.search(pattern, content, re.MULTILINE)
    if not match:
        return ""

    start = match.end()
    next_heading = re.search(r"\n=+ ", content[start:])
    if next_heading:
        section_text = content[start:start + next_heading.start()]
    else:
        section_text = content[start:]

    section_text = re.sub(r"//.*", "", section_text)
    section_text = re.sub(r"#[^\s({]+(\([^)]*\))?(\{[^}]*\})?", "", section_text)
    section_text = re.sub(r"\*+", "", section_text)
    section_text = re.sub(r"\[+\d+\]", "", section_text)
    section_text = re.sub(r"\n\s*\n", "\n", section_text).strip()
    return section_text


def print_result(result: dict, threshold_pct: float = 1.0):
    if not result.get("ok"):
        print(f"ERROR: {result.get('error', 'Unknown error')}")
        return False

    stats = result.get("results", {}).get("statistics", {})
    res_info = result.get("results", {}).get("res_info", {})

    score = stats.get("score", 0)
    score_pct = score * 100

    print(f"  Segments: {res_info.get('total', '?')}")
    print(f"  AI score: {score_pct:.4f}%")

    if score_pct <= 1:
        print("  Verdict: ✅ PASS (≤1%)")
    elif score_pct <= 5:
        print(f"  Verdict: ⚠️  BORDERLINE ({score_pct:.2f}%)")
    else:
        print(f"  Verdict: ❌ FAIL ({score_pct:.2f}% — needs rewrite)")

    return score_pct <= threshold_pct


def main():
    parser = argparse.ArgumentParser(description="Check AI score via Yandex Neurodetector")
    parser.add_argument("--text", type=str, help="Text to check")
    parser.add_argument("--file", type=str, help="File to read text from")
    parser.add_argument("--pdf", type=str, help="PDF file to check (full-doc analysis)")
    parser.add_argument("--typ", type=str, help=".typ file to extract section from")
    parser.add_argument("--section", type=str, help="Section title to extract (with --typ)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help=f"AI score threshold in percent (default {1.0})")

    args = parser.parse_args()

    text = None
    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    elif args.pdf:
        print(f"\n=== PDF: {args.pdf} ===")
        result = check_pdf(args.pdf)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        passed = print_result(result, args.threshold)
        sys.exit(0 if passed else 1)
        return
    elif args.typ and args.section:
        text = extract_section_from_typ(args.typ, args.section)
        if not text:
            print(f"ERROR: Section '{args.section}' not found in {args.typ}")
            sys.exit(1)
        print(f"\n=== Section: {args.section} ({len(text)} chars) ===")
    else:
        text = sys.stdin.read().strip()

    if not text:
        print("ERROR: No text provided")
        sys.exit(1)

    result = check_text(text)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    passed = print_result(result, args.threshold)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
