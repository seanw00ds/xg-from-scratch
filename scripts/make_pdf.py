"""Render report/A2_report.md to the submission PDF via headless Chrome."""

import base64
import re
import subprocess
import sys
from pathlib import Path

import markdown

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
SRC = Path("report/A2_report.md")
HTML = Path("report/A2_report.html")
PDF = Path("report/SeanWoods_26107565_2026_UTS_ML_Journal.pdf")

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: Charter, Georgia, 'Times New Roman', serif;
       font-size: 10.5pt; line-height: 1.45; color: #111; max-width: 100%; }
h1 { font-size: 19pt; margin: 0 0 2pt; }
h2 { font-size: 13.5pt; margin: 20pt 0 6pt; border-bottom: 1px solid #bbb;
     padding-bottom: 3pt; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 14pt 0 4pt; page-break-after: avoid; }
h1 + h3 { font-weight: normal; font-style: italic; color: #444; margin-top: 0; }
p, li { text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 10pt 0;
        font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: 3pt 5pt; text-align: left; }
th { background: #f2f2f2; }
code { font-family: 'SF Mono', Menlo, Consolas, monospace; font-size: 9pt;
       background: #f5f5f5; padding: 0 2px; }
pre { background: #f6f6f6; border-left: 3px solid #bbb; padding: 7pt 9pt;
      font-size: 9pt; overflow-x: auto; page-break-inside: avoid; }
pre code { background: none; }
blockquote { margin: 10pt 0; padding: 6pt 12pt; border-left: 3px solid #888;
             background: #fafafa; font-style: italic; }
img { max-width: 92%; display: block; margin: 10pt auto 2pt; }
img + em, p > em:only-child { display: block; text-align: center;
                              font-size: 8.5pt; color: #555; margin-bottom: 12pt; }
hr { border: none; border-top: 1px solid #ddd; margin: 16pt 0; }
"""


def embed_images(html):
    """Inline every figure as base64 so the PDF is one self-contained file."""
    def repl(m):
        src = m.group(1)
        path = (SRC.parent / src).resolve()
        if not path.exists():
            print(f"  missing image: {src}", file=sys.stderr)
            return m.group(0)
        b64 = base64.b64encode(path.read_bytes()).decode()
        return f'src="data:image/png;base64,{b64}"'
    return re.sub(r'src="([^"]+)"', repl, html)


def main():
    text = SRC.read_text()
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    body = embed_images(body)
    HTML.write_text(f"<!doctype html><html><head><meta charset='utf-8'>"
                    f"<style>{CSS}</style></head><body>{body}</body></html>")

    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF.resolve()}", HTML.resolve().as_uri()],
                   check=True, capture_output=True)
    print(f"wrote {PDF} ({PDF.stat().st_size / 1024:.0f} KB)")

    if "PASTE_COLAB_URL_HERE" in text or "PASTE_GITHUB_URL_HERE" in text:
        print("\n  WARNING: the notebook and repo URLs are still placeholders.")


if __name__ == "__main__":
    main()
