#!/usr/bin/env python3
"""Render .mmd files to PNG via mermaid.ink API (returns JPEG, converted to PNG)."""
import base64, sys, os, urllib.request, urllib.error, time
from io import BytesIO
from PIL import Image

INK_URL = "https://mermaid.ink/img/{encoded}"

def mmd_to_png(mmd_text: str) -> bytes | None:
    encoded = base64.urlsafe_b64encode(mmd_text.encode('utf-8')).decode('utf-8').rstrip('=')
    url = INK_URL.format(encoded=encoded)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            jpeg_data = resp.read()
        # mermaid.ink returns JPEG, convert to PNG
        img = Image.open(BytesIO(jpeg_data))
        buf = BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        return None

def main():
    files = [f for f in sys.argv[1:] if f.endswith('.mmd')]
    if not files:
        print("Usage: render_mermaid.py *.mmd", file=sys.stderr)
        sys.exit(1)
    for f in files:
        png_path = f.replace('.mmd', '.png')
        print(f"Rendering {f} -> {png_path} ...")
        with open(f) as fh:
            mmd = fh.read()
        png = mmd_to_png(mmd)
        if png:
            with open(png_path, 'wb') as fh:
                fh.write(png)
            size = os.path.getsize(png_path)
            print(f"  OK: {size/1024:.1f} KB")
        else:
            print(f"  FAILED", file=sys.stderr)
        time.sleep(0.5)

if __name__ == '__main__':
    main()
