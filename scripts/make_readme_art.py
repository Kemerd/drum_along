"""
make_readme_art.py - draw the README graphics as self-contained SVG files.

Usage:
    python scripts/make_readme_art.py

Writes docs/img/hero.svg, docs/img/how-it-works.svg and docs/img/results.svg.

Every graphic is a dark rounded card, so it reads the same on GitHub's light and
dark themes. Motion is plain CSS inside the SVG, which GitHub keeps when the file
is shown through an <img> tag, and it switches off for readers who ask for
reduced motion. Result figures come from docs/NOTES.md.
"""

import math
import random
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, Roboto, Helvetica, Arial, sans-serif"
INK = "#f0efec"          # primary text
INK2 = "#c3c2b7"         # secondary text
INK3 = "#8f8e86"         # muted text
SURFACE = "#1a1a19"      # card
RAISED = "#262624"       # panels on the card
LINE = "#3a3a37"         # hairlines
BLUE = "#3987e5"         # after / kick
ORANGE = "#d95926"       # before / snare
AQUA = "#199e70"         # hats

BEAT_SECONDS = 2.0       # one bar loop, 120 BPM


def card(width: int, height: int, body: str, style: str = "") -> str:
    """Wrap drawing code in a rounded dark card with the shared motion rules."""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">
<style>
  text {{ font-family: {FONT}; }}
  {style}
  @media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; }} }}
</style>
<rect width="{width}" height="{height}" rx="24" fill="{SURFACE}"/>
{body}
</svg>
"""


# ===========================================================================
# Hero: sound goes in, a real groove comes out
# ===========================================================================

def hero() -> str:
    width, height = 1280, 344
    style = f"""
  @keyframes hit {{ 0% {{ opacity: 1; transform: scale(1.22); }} 18% {{ opacity: 1; transform: scale(1); }} 45%, 100% {{ opacity: .42; transform: scale(1); }} }}
  @keyframes sweep {{ from {{ transform: translateX(0); }} to {{ transform: translateX(576px); }} }}
  @keyframes wave {{ 0%, 100% {{ transform: scaleY(.45); }} 12% {{ transform: scaleY(1); }} }}
  .hit {{ transform-box: fill-box; transform-origin: center; animation: hit {BEAT_SECONDS}s linear infinite; }}
  .head {{ animation: sweep {BEAT_SECONDS}s linear infinite; }}
  .bar {{ transform-box: fill-box; transform-origin: center; animation: wave {BEAT_SECONDS / 2}s ease-out infinite; }}
"""
    parts = [f"""<defs>
  <radialGradient id="glow" cx="72%" cy="50%" r="60%">
    <stop offset="0" stop-color="{BLUE}" stop-opacity=".20"/>
    <stop offset="1" stop-color="{BLUE}" stop-opacity="0"/>
  </radialGradient>
</defs>
<rect width="{width}" height="{height}" rx="24" fill="url(#glow)"/>
<text x="72" y="150" font-size="84" font-weight="700" letter-spacing="-2.5" fill="{INK}">drum_along</text>
<text x="74" y="204" font-size="32" font-weight="500" fill="{INK2}">You play. It drums.</text>
<text x="74" y="246" font-size="20" fill="{INK3}">No click. No BPM box. No judgement.</text>"""]

    # Incoming sound: one thin bar per slice, pulsing twice a bar.
    rng = random.Random(7)
    left, top, cell, gap = 620, 74, 30, 6
    grid_width = 16 * (cell + gap) + 3 * 10 - gap
    parts.append(f'<text x="{left}" y="{top - 28}" font-size="14" font-weight="600" letter-spacing="1.6" fill="{INK3}">YOU</text>')
    bars = 72
    for i in range(bars):
        x = left + i * (grid_width / bars)
        beat = (i / bars * 8) % 1.0
        h = 8 + 34 * math.exp(-3.2 * beat) * (0.55 + 0.45 * rng.random()) + 6 * rng.random()
        delay = (i % 18) / 18 * (BEAT_SECONDS / 2)
        parts.append(f'<rect class="bar" style="animation-delay:{delay:.3f}s" x="{x:.1f}" y="{top + 2 - h / 2:.1f}" '
                     f'width="3.2" height="{h:.1f}" rx="1.6" fill="{INK3}"/>')

    # The groove itself: the stock rock pattern, one bar, sixteenths.
    lanes = [("HATS", "x.x.x.x.x.x.x.x.", AQUA), ("SNARE", "....x.......x...", ORANGE), ("KICK", "x.....x.x.x.....", BLUE)]
    grid_top = top + 68
    parts.append(f'<text x="{left}" y="{grid_top - 12}" font-size="14" font-weight="600" letter-spacing="1.6" fill="{INK3}">IT</text>')

    def column_x(step: int) -> float:
        return left + step * (cell + gap) + (step // 4) * 10

    for row, (_, pattern, color) in enumerate(lanes):
        y = grid_top + row * (cell + gap + 4)
        for step, mark in enumerate(pattern):
            x = column_x(step)
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="7" fill="{RAISED}"/>')
            if mark == "x":
                delay = step / 16 * BEAT_SECONDS
                parts.append(f'<rect class="hit" style="animation-delay:{delay:.3f}s" x="{x}" y="{y}" '
                             f'width="{cell}" height="{cell}" rx="7" fill="{color}"/>')
    grid_height = 3 * (cell + gap + 4) - gap - 4
    parts.append(f'<rect class="head" x="{left - 3}" y="{grid_top - 8}" width="3" height="{grid_height + 16}" rx="1.5" fill="{INK}" opacity=".85"/>')

    labels = "".join(f'<text x="{left - 16}" y="{grid_top + row * (cell + gap + 4) + 21}" text-anchor="end" font-size="12" '
                     f'font-weight="600" letter-spacing="1.2" fill="{INK3}">{name}</text>'
                     for row, (name, _, _) in enumerate(lanes))
    parts.append(labels)
    return card(width, height, "\n".join(parts), style)


# ===========================================================================
# How it works: one line of boxes, with signal running through them
# ===========================================================================

def how_it_works() -> str:
    width, height = 1280, 372
    style = """
  @keyframes flow { from { offset-distance: 0%; opacity: 0; } 15% { opacity: 1; } 85% { opacity: 1; } to { offset-distance: 100%; opacity: 0; } }
"""
    nodes = [
        ("You", "AUDIO IN", ["Guitar, piano or a", "whole band. Plain", "sound, no BPM", "and no click."], INK3),
        ("Ears", "TRAINED", ["A neural net hears", "the beat, the 1, and", "if a drummer belongs.", "510k weights."], BLUE),
        ("Clock", "HAND-WRITTEN", ["Tempo and phase.", "Bends with you, holds", "through a riff, waits", "out a stop."], AQUA),
        ("Drummer", "HAND-TUNED", ["Grooves, fills and", "dynamics. All of it", "in a JSON file you", "can edit."], ORANGE),
        ("Your drums", "MIDI OUT", ["General MIDI into", "Kontakt, Drum Rack", "or any drum plugin", "you like."], INK3),
    ]
    box_w, box_h, gap, top = 212, 236, 26, 68
    left = (width - (5 * box_w + 4 * gap)) / 2
    parts = []
    for i, (title, tag, lines, color) in enumerate(nodes):
        x = left + i * (box_w + gap)
        parts.append(f'<rect x="{x}" y="{top}" width="{box_w}" height="{box_h}" rx="16" fill="{RAISED}"/>')
        parts.append(f'<rect x="{x + 22}" y="{top + 24}" width="30" height="4" rx="2" fill="{color}"/>')
        parts.append(f'<text x="{x + 22}" y="{top + 62}" font-size="27" font-weight="700" letter-spacing="-.4" fill="{INK}">{title}</text>')
        parts.append(f'<text x="{x + 22}" y="{top + 86}" font-size="11.5" font-weight="600" letter-spacing="1.3" fill="{INK3}">{tag}</text>')
        for j, line in enumerate(lines):
            parts.append(f'<text x="{x + 22}" y="{top + 122 + j * 23}" font-size="15.5" fill="{INK2}">{line}</text>')
        if i < 4:
            y = top + box_h / 2
            x0, x1 = x + box_w + 3, x + box_w + gap - 3
            parts.append(f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{LINE}" stroke-width="2" stroke-linecap="round"/>')
            parts.append(f'<circle r="4" fill="{INK}" style="offset-path: path(\'M {x0} {y} L {x1} {y}\'); '
                         f'animation: flow 1s linear infinite; animation-delay: {i * 0.2:.1f}s"/>')
    parts.append(f'<text x="{width / 2}" y="{top + box_h + 40}" text-anchor="middle" font-size="16" fill="{INK3}">'
                 f'Notes are scheduled ahead of the beat, so the drums land on time instead of reacting late.</text>')
    return card(width, height, "\n".join(parts), style)


# ===========================================================================
# Results: before and after on one percent axis, stumbles as their own number
# ===========================================================================

def results() -> str:
    width, height = 1280, 470
    rows = [
        ("Beats on the real beat", 59, 84),
        ("Right tempo", 63, 85),
        ("Snare on the right beats", 38, 78),
        ("The 1 on the real 1", 25, 69),
    ]
    label_x, axis_x, axis_w = 64, 330, 520
    top, row_h = 164, 58

    def px(value: float) -> float:
        return axis_x + axis_w * value / 100.0

    parts = [
        f'<text x="64" y="72" font-size="30" font-weight="700" letter-spacing="-.5" fill="{INK}">119 songs it had never heard</text>',
        f'<text x="64" y="102" font-size="16" fill="{INK2}">How often it got it right, against exact MIDI.</text>',
    ]

    # Legend: two series, hollow versus solid, so identity is not color alone.
    legend_y = 132
    parts.append(f'<circle cx="{axis_x + 8}" cy="{legend_y}" r="7" fill="{SURFACE}" stroke="{ORANGE}" stroke-width="2.5"/>')
    parts.append(f'<text x="{axis_x + 24}" y="{legend_y + 5}" font-size="14" fill="{INK2}">Hand-written only</text>')
    parts.append(f'<circle cx="{axis_x + 190}" cy="{legend_y}" r="8" fill="{BLUE}"/>')
    parts.append(f'<text x="{axis_x + 206}" y="{legend_y + 5}" font-size="14" fill="{INK2}">With the trained model</text>')

    # Recessive grid, then one dumbbell per measure.
    bottom = top + row_h * len(rows)
    for tick in (0, 25, 50, 75, 100):
        x = px(tick)
        parts.append(f'<line x1="{x}" y1="{top - 4}" x2="{x}" y2="{bottom - 18}" stroke="{LINE}" stroke-width="1"/>')
        parts.append(f'<text x="{x}" y="{bottom + 4}" text-anchor="middle" font-size="13" fill="{INK3}">{tick}%</text>')
    for i, (name, before, after) in enumerate(rows):
        y = top + 18 + i * row_h
        parts.append(f'<text x="{label_x}" y="{y + 6}" font-size="17" fill="{INK}">{name}</text>')
        parts.append(f'<line x1="{px(before)}" y1="{y}" x2="{px(after)}" y2="{y}" stroke="{BLUE}" stroke-width="2"/>')
        parts.append(f'<circle cx="{px(before)}" cy="{y}" r="8" fill="{SURFACE}" stroke="{ORANGE}" stroke-width="2.5"/>')
        parts.append(f'<circle cx="{px(after)}" cy="{y}" r="9" fill="{BLUE}"/>')
        parts.append(f'<text x="{px(before) - 16}" y="{y + 5}" text-anchor="end" font-size="15" fill="{INK2}">{before}%</text>')
        parts.append(f'<text x="{px(after) + 16}" y="{y + 5}" font-size="15" font-weight="700" fill="{INK}">{after}%</text>')

    # Stumbles have their own scale, so they get their own tile, not a second axis.
    tile_x, tile_y, tile_w, tile_h = 950, 64, 266, 342
    parts.append(f'<rect x="{tile_x}" y="{tile_y}" width="{tile_w}" height="{tile_h}" rx="16" fill="{RAISED}"/>')
    parts.append(f'<text x="{tile_x + 28}" y="{tile_y + 46}" font-size="17" font-weight="600" fill="{INK}">Stumbles per minute</text>')
    parts.append(f'<text x="{tile_x + 28}" y="{tile_y + 72}" font-size="14" fill="{INK2}">The clock jumping mid-song</text>')
    parts.append(f'<text x="{tile_x + 28}" y="{tile_y + 196}" font-size="88" font-weight="700" letter-spacing="-3" fill="{INK}">0.1</text>')
    parts.append(f'<text x="{tile_x + 28}" y="{tile_y + 244}" font-size="17" fill="{INK2}">down from 4.5</text>')
    parts.append(f'<text x="{tile_x + 28}" y="{tile_y + 300}" font-size="14" fill="{INK3}">About one every 10 minutes.</text>')

    parts.append(f'<text x="64" y="{height - 28}" font-size="14" fill="{INK3}">Slakh2100 test songs, drums removed. No singing in this data, so real records are a harder test.</text>')
    return card(width, height, "\n".join(parts))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, svg in (("hero.svg", hero()), ("how-it-works.svg", how_it_works()), ("results.svg", results())):
        (OUT / name).write_text(svg, encoding="utf-8")
        print(f"wrote {OUT / name} ({len(svg) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
