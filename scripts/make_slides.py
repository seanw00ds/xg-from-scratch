"""Render report/slides.html to the A3 presentation PDF via headless Chrome.

slides.html references the figures in results/ by relative path. Chrome will not
load those from a file:// page reliably, so the figures are inlined as base64
into a throwaway copy, which is gitignored: slides.html stays the only source.
"""

import base64
import re
import subprocess
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
SRC = Path("report/slides.html")
TMP = Path("report/slides_embedded.html")
PDF = Path("report/SeanWoods_A3_slides.pdf")


def embed(html):
    def repl(m):
        path = (SRC.parent / m.group(1)).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return f'src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}"'
    return re.sub(r'src="(\.\./results/[^"]+)"', repl, html)


def main():
    TMP.write_text(embed(SRC.read_text()))
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF.resolve()}", TMP.resolve().as_uri()],
                   check=True, capture_output=True)
    TMP.unlink()
    print(f"wrote {PDF} ({PDF.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
