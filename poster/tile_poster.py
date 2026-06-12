#!/usr/bin/env python3
"""Split poster.pdf (36x24 in) into a 2x2 grid of 11x17-landscape sheets.

Each sheet carries one quadrant plus an overlap strip on interior seams,
with crop marks and a dashed cut line on the edges that must be trimmed.
Assembled size: ~30.6 x 20.4 in (85% of 36x24, same aspect ratio).

Usage: python3 tile_poster.py [poster.pdf] [poster_tiled_11x17.pdf]
"""
import sys
import fitz

SRC = sys.argv[1] if len(sys.argv) > 1 else "poster.pdf"
OUT = sys.argv[2] if len(sys.argv) > 2 else "poster_tiled_11x17.pdf"

SHEET_W, SHEET_H = 17 * 72, 11 * 72   # 11x17 landscape, in points
MARGIN = 0.30 * 72                    # unprintable border kept clear
OVERLAP = 0.40 * 72                   # duplicated strip at interior seams

src = fitz.open(SRC)
page = src[0]
PW, PH = page.rect.width, page.rect.height          # 2592 x 1728

printable_w = SHEET_W - 2 * MARGIN
printable_h = SHEET_H - 2 * MARGIN
cover_w = 2 * printable_w - OVERLAP                  # assembled max area
cover_h = 2 * printable_h - OVERLAP
scale = min(cover_w / PW, cover_h / PH)              # fit, keep aspect
ov_src = OVERLAP / scale                             # overlap in source pts

# centered seams: each tile covers half the poster + half the overlap
xs = [(0, PW / 2 + ov_src / 2), (PW / 2 - ov_src / 2, PW)]
ys = [(0, PH / 2 + ov_src / 2), (PH / 2 - ov_src / 2, PH)]

names = {(0, 0): "TOP-LEFT", (1, 0): "TOP-RIGHT",
         (0, 1): "BOTTOM-LEFT", (1, 1): "BOTTOM-RIGHT"}
GRAY = (0.45, 0.45, 0.45)

out = fitz.open()
n = 0
for j, (sy0, sy1) in enumerate(ys):          # rows
    for i, (sx0, sx1) in enumerate(xs):      # cols
        n += 1
        sheet = out.new_page(width=SHEET_W, height=SHEET_H)
        w = (sx1 - sx0) * scale
        h = (sy1 - sy0) * scale
        tgt = fitz.Rect(MARGIN, MARGIN, MARGIN + w, MARGIN + h)
        sheet.show_pdf_page(tgt, src, 0, clip=fitz.Rect(sx0, sy0, sx1, sy1))

        # crop marks just outside the content corners
        for cx in (tgt.x0, tgt.x1):
            for cy in (tgt.y0, tgt.y1):
                sheet.draw_line(fitz.Point(cx - 10 if cx == tgt.x0 else cx + 2, cy),
                                fitz.Point(cx - 2 if cx == tgt.x0 else cx + 10, cy),
                                color=GRAY, width=0.4)
                sheet.draw_line(fitz.Point(cx, cy - 10 if cy == tgt.y0 else cy + 2),
                                fitz.Point(cx, cy - 2 if cy == tgt.y0 else cy + 10),
                                color=GRAY, width=0.4)

        # dashed cut line on interior edges (right column: cut LEFT; bottom row: cut TOP)
        cuts = []
        if i == 1:
            cuts.append("LEFT")
            sheet.draw_line(fitz.Point(tgt.x0 - 1.5, tgt.y0), fitz.Point(tgt.x0 - 1.5, tgt.y1),
                            color=GRAY, width=0.4, dashes="[4 3] 0")
        if j == 1:
            cuts.append("TOP")
            sheet.draw_line(fitz.Point(tgt.x0, tgt.y0 - 1.5), fitz.Point(tgt.x1, tgt.y0 - 1.5),
                            color=GRAY, width=0.4, dashes="[4 3] 0")

        note = f"Sheet {n}/4 - {names[(i, j)]}"
        note += f" - cut white margin along dashed {' & '.join(cuts)} edge(s), lay over neighbor" if cuts else " - do not trim"
        sheet.insert_text(fitz.Point(MARGIN, SHEET_H - 7), note, fontsize=7, color=GRAY)

out.save(OUT)
asm_w, asm_h = PW * scale / 72, PH * scale / 72
print(f"wrote {OUT}: 4 sheets of {SHEET_W/72:.0f}x{SHEET_H/72:.0f} in (landscape)")
print(f"assembled poster: {asm_w:.1f} x {asm_h:.1f} in  (scale {scale*100:.0f}% of 36x24)")
print(f"overlap strip: {OVERLAP/72:.2f} in on interior seams")
