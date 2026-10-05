"""
Build-time only: render branding/logo.svg into the app icons.

  branding/lumina.ico      Windows: installer exe, Start Menu, Add/Remove Programs
  branding/lumina-256.png  Linux: icon theme / desktop entry, installer window

Sizes up to 48 px drop the wordmark, which is unreadable noise that small.
Needs playwright (with Chrome) and Pillow:  python Installer/make_icon.py
"""
import io
import re
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
BRAND = ROOT / "branding"


def render(svg: str, px: int = 512) -> Image.Image:
    html = f"<html><body style='margin:0;background:transparent'>{svg.replace('width=\"500\" height=\"500\"', f'width=\"{px}\" height=\"{px}\"')}</body></html>"
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome")
        page = b.new_page(viewport={"width": px, "height": px})
        page.set_content(html)
        png = page.screenshot(omit_background=True)
        b.close()
    return Image.open(io.BytesIO(png)).convert("RGBA")


svg = (BRAND / "logo.svg").read_text(encoding="utf-8")
full = render(svg)
bare = render(re.sub(r"<text[\s\S]*?</text>", "", svg))

big = {s: full.resize((s, s), Image.LANCZOS) for s in (64, 128, 256)}
small = {s: bare.resize((s, s), Image.LANCZOS) for s in (16, 20, 24, 32, 40, 48)}
frames = {**small, **big}

big[256].save(BRAND / "lumina-256.png")
big[256].save(BRAND / "lumina.ico", sizes=[(s, s) for s in frames],
              append_images=[frames[s] for s in frames if s != 256])
print("wrote", BRAND / "lumina.ico", "and", BRAND / "lumina-256.png")
