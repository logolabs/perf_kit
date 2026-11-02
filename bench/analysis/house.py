"""LogoLabs house style for the perf_kit figures: the Agate release sheets (agate-preview-003) in the colours of
the perf_kit masthead (assets/masthead.png). Red and black on warm paper, mono caps labels, hairline rules, a
heavy grotesk headline that states the finding, a mono footer naming the data source.

Every figure is an HTML page with inline SVG, rendered by headless Chromium (playwright; Microsoft Edge when
installed, else playwright's own Chromium) and screenshotted at 2x. The pages are written next to this file in
html/ so they can be opened and inspected as they are.

    pip install playwright && playwright install chromium     # once, if Edge is not installed
"""
from __future__ import annotations

import html
from pathlib import Path

HERE = Path(__file__).resolve().parent
HTML_DIR = HERE / "html"

# masthead paper, ink and red; greys for the baselines, darkest = strongest baseline
PAPER, INK, RED, MUTE, FAINT, GRID = "#FEFEF8", "#0B0A08", "#F92811", "#6E6A62", "#D9D6CE", "#E2DED5"
GREYS = ("#0B0A08", "#5A564F", "#8E897F", "#B9B4AA")

FONTS = ("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500"
         "&family=Inter+Tight:wght@600;700;800&family=Playfair+Display:wght@700&display=block")
CSS = f"""
:root{{--paper:{PAPER};--ink:{INK};--red:{RED};--mute:{MUTE};--faint:{FAINT}}}
*{{box-sizing:border-box;margin:0;padding:0}}
html,body{{background:var(--paper)}}
body{{color:var(--ink);font-family:'IBM Plex Sans',sans-serif;-webkit-font-smoothing:antialiased}}
.sheet{{padding:30px 40px 26px;width:max-content;background:var(--paper)}}
.mono{{font-family:'IBM Plex Mono',monospace;letter-spacing:.14em;text-transform:uppercase}}
.num{{font-family:'IBM Plex Mono',monospace;font-variant-numeric:tabular-nums}}
.top{{display:flex;justify-content:space-between;align-items:center;font:500 13px 'IBM Plex Mono';letter-spacing:.16em;
     text-transform:uppercase;padding-bottom:12px;border-bottom:1.5px solid var(--ink)}}
.top .r{{color:var(--red)}}
h1{{font:700 46px/1.02 'Inter Tight';letter-spacing:-.03em;margin:26px 0 8px}}
.sub{{font:300 18px/1.4 'IBM Plex Sans';color:#26241F;margin-bottom:24px}}
.foot{{display:flex;justify-content:space-between;align-items:center;gap:24px;border-top:1.5px solid var(--ink);margin-top:22px;
      padding-top:11px;font:400 12px 'IBM Plex Mono';letter-spacing:.1em;text-transform:uppercase;color:var(--mute)}}
.sq{{width:14px;height:14px;background:var(--ink);flex:none}}
.red{{color:var(--red)}}
.legend{{display:flex;flex-wrap:wrap;gap:10px 28px;font:400 12px 'IBM Plex Mono';letter-spacing:.1em;text-transform:uppercase;
        margin-bottom:14px;align-items:center}}
.legend>span{{display:flex;align-items:center;gap:8px}}
.axis{{font:400 12.5px 'IBM Plex Mono';color:var(--mute);letter-spacing:.06em}}
.axcap{{font:400 12px 'IBM Plex Mono';letter-spacing:.12em;text-transform:uppercase}}
"""


def esc(s: str) -> str:
    return html.escape(s, quote=False)


def lockup(height: int = 30) -> str:
    """The LogoLabs mark and name, as at the top left of the masthead."""
    svg = (HERE / "logolabs-mark.svg").read_text(encoding="utf-8")
    svg = svg.replace("<svg ", f"<svg style='height:{height}px;width:auto;display:block;color:var(--ink)' ", 1)
    return (f"<span style='display:flex;align-items:center;gap:10px;text-transform:none;letter-spacing:0'>{svg}"
            f"<span style=\"font:700 24px 'Playfair Display';color:var(--ink)\">LogoLabs</span></span>")


def page(body: str, top_left: str, top_right: str, foot: str, width: int | None = None) -> str:
    w = f"width:{width}px;" if width else ""
    return (f'<!doctype html>\n<html><head><meta charset="utf-8"><link rel="stylesheet" href="{FONTS}">'
            f"<style>{CSS}</style></head><body><div class='sheet' id='sheet' style='{w}'>"
            f"<div class='top'><span>{top_left}</span><span class='r'>{top_right}</span></div>{body}"
            f"<div class='foot'><span>{foot}</span><span class='sq'></span></div></div></body></html>\n")


def _browser(p):
    try:
        return p.chromium.launch(channel="msedge")
    except Exception:  # noqa: BLE001 - no Edge on this machine: playwright's bundled Chromium
        return p.chromium.launch()


def render(jobs: list[tuple[str, Path]], scale: float = 2.0) -> None:
    """[(html document, output png)] -> each page's #sheet screenshotted at `scale`; the HTML is kept in html/."""
    from playwright.sync_api import sync_playwright
    HTML_DIR.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = _browser(p)
        for doc, out in jobs:
            f = HTML_DIR / (out.stem + ".html")
            f.write_bytes(doc.encode("utf-8"))            # LF only, no BOM, on every OS
            pg = browser.new_page(viewport={"width": 800, "height": 600}, device_scale_factor=scale)
            pg.goto(f.as_uri())
            pg.wait_for_load_state("networkidle")
            pg.evaluate("document.fonts.ready")
            out.parent.mkdir(parents=True, exist_ok=True)
            pg.locator("#sheet").screenshot(path=str(out))
            pg.close()
            print("rendered", out)
        browser.close()


def print_pdf(html_path: Path, pdf_path: Path, foot_text: str) -> None:
    """Print an HTML report to A4 PDF the way the LogoLabs reports are printed (headless Chromium, mono footer
    with page numbers, 16 mm top and bottom margins, backgrounds on)."""
    from playwright.sync_api import sync_playwright
    foot = ('<div style="width:100%;padding:0 17mm;font:7.5px IBM Plex Mono, Consolas, monospace;letter-spacing:.12em;'
            f'text-transform:uppercase;color:{MUTE};display:flex;justify-content:space-between">'
            f'<span>{foot_text}</span>'
            '<span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>')
    with sync_playwright() as p:
        b = _browser(p)
        pg = b.new_page()
        pg.goto(html_path.resolve().as_uri())
        pg.wait_for_load_state("networkidle")
        pg.evaluate("document.fonts.ready")
        pg.wait_for_timeout(600)
        pg.pdf(path=str(pdf_path), format="A4", print_background=True, prefer_css_page_size=True,
               display_header_footer=True, header_template="<div></div>", footer_template=foot,
               margin={"top": "16mm", "bottom": "16mm", "left": "0mm", "right": "0mm"})
        b.close()
