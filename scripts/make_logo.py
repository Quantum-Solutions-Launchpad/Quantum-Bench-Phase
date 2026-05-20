"""Generate the QuaPh favicon / navbar mark from the console banner.

The mark uses a square viewBox over just the "Q" slice of the banner.
Each non-space character in the slice emits an oversized `<rect>` so the
sparse ASCII strokes fuse into a continuous silhouette, which is then
filled with the same cyan -> magenta gradient the rich console uses.
The all-rects implementation has no font or filter dependencies, so it
rasterizes crisply at every size -- including 16-32 px favicons.

Outputs (under docs/_static/):

  favicon-light.svg  filled Q silhouette, light theme
  favicon-dark.svg   filled Q silhouette, dark theme
"""

from __future__ import annotations

from pathlib import Path

from quaph._console import _BANNER


DARK_GRADIENT = ("#00D9FF", "#7F9CE8", "#FF5FD2")
LIGHT_GRADIENT = ("#00A8C8", "#7060C8", "#D43FA8")

# Column slice that contains the "Q" glyph in the banner.
Q_COL_START = 0
Q_COL_END = 16

# Character cell dimensions in user-space units.
CHAR_W = 10.0
LINE_H = 12.0

# How much each cell rect overlaps its neighbours; > 1 fuses adjacent
# occupied cells into a continuous silhouette.
BLEED = 1.30


def _q_lines() -> list[str]:
    lines = _BANNER.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    sliced = [line[Q_COL_START:Q_COL_END].rstrip() for line in lines]
    while sliced and not sliced[0].strip():
        sliced.pop(0)
    while sliced and not sliced[-1].strip():
        sliced.pop()
    return sliced


def build_svg(lines: list[str], gradient: tuple[str, str, str]) -> str:
    if not lines:
        raise ValueError("no banner lines to render")

    max_cols = max(len(line) for line in lines)
    n_lines = len(lines)
    content_w = max_cols * CHAR_W
    content_h = n_lines * LINE_H

    side = max(content_w, content_h)
    ox = (side - content_w) / 2.0
    oy = (side - content_h) / 2.0
    over = max(CHAR_W, LINE_H) * (BLEED - 1.0) / 2.0
    vb_w = vb_h = side + 2 * over
    vb_x = -over - ox
    vb_y = -over - oy

    w = CHAR_W * BLEED
    h = LINE_H * BLEED
    dx = (w - CHAR_W) / 2.0
    dy = (h - LINE_H) / 2.0

    rects: list[str] = []
    for row, line in enumerate(lines):
        for col, ch in enumerate(line):
            if ch == " ":
                continue
            x = col * CHAR_W - dx
            y = row * LINE_H - dy
            rects.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" '
                f'width="{w:.2f}" height="{h:.2f}" fill="white"/>'
            )

    c0, c1, c2 = gradient
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{vb_x:.2f} {vb_y:.2f} {vb_w:.2f} {vb_h:.2f}" '
        'role="img" aria-label="QuaPh">'
        "<defs>"
        '<linearGradient id="qph-grad" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{c0}"/>'
        f'<stop offset="50%" stop-color="{c1}"/>'
        f'<stop offset="100%" stop-color="{c2}"/>'
        "</linearGradient>"
        '<mask id="qph-mask" maskUnits="userSpaceOnUse" '
        f'x="{vb_x:.2f}" y="{vb_y:.2f}" '
        f'width="{vb_w:.2f}" height="{vb_h:.2f}">'
        + "".join(rects)
        + "</mask>"
        "</defs>"
        f'<rect x="{vb_x:.2f}" y="{vb_y:.2f}" '
        f'width="{vb_w:.2f}" height="{vb_h:.2f}" '
        'fill="url(#qph-grad)" mask="url(#qph-mask)"/>'
        "</svg>"
    )


def main() -> None:
    here = Path(__file__).resolve().parent
    static = here.parent / "docs" / "_static"
    static.mkdir(parents=True, exist_ok=True)

    lines = _q_lines()
    (static / "favicon-dark.svg").write_text(build_svg(lines, DARK_GRADIENT))
    (static / "favicon-light.svg").write_text(build_svg(lines, LIGHT_GRADIENT))

    for name in ("favicon-light.svg", "favicon-dark.svg"):
        print(f"Wrote {static / name}")


if __name__ == "__main__":
    main()
