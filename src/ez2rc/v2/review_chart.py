"""Self-contained SVG chart view for local draft review."""
from fractions import Fraction
from html import escape
import math

from .chart_layout import COLUMN_MEASURES, KEY_COLORS, LANE_FILLS, MEASURE_RAIL, chart_columns, number_rail_width
from .contracts import read_ratio


def chart_view(draft):
    notes = draft['notes']
    measure = draft['projection'].get('measure_beats')
    measure = read_ratio(measure) if measure is not None else None
    if measure is not None and measure <= 0:
        measure = None
    span = measure * COLUMN_MEASURES if measure else Fraction(16)
    lanes = draft['metadata']['lane_count']
    height, top, scale = 600, 34, 540 / float(span)
    blocks = {}
    missing = []
    for index, note in enumerate(notes):
        start = read_ratio(note['start_beat']) if note['start_beat'] is not None else None
        end = read_ratio(note['end_beat']) if note['end_beat'] is not None else start
        if start is None:
            missing.append(f'<li><a href="#note-{index}">{escape(note["id"])}</a> · Lane {note["lane"]+1} · Start beat needs review</li>')
            continue
        end = end if end is not None else start
        first = math.floor(start / span)
        last = math.ceil(end / span) - 1 if end > start else first
        # Bound rendering of malformed or exceptionally long manual notes.
        if last - first > 256:
            missing.append(f'<li><a href="#note-{index}">{escape(note["id"])}</a> · Hold exceeds the display range</li>')
            continue
        for block in range(first, last + 1):
            blocks.setdefault(block, []).append((index, note, start, end))
    rail_width = number_rail_width(blocks, measure)
    rail_left = 4
    lane_left = rail_left + rail_width
    lane_right = lane_left + lanes * 34
    width = lane_right + 4
    panels = []
    for block in chart_columns(blocks):
        base = block * span

        def y(beat):
            return top + float(base + span - beat) * scale

        pieces = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Chart starting at beat {float(base):g}">',
                  f'<rect x="{rail_left}" y="{top}" width="{rail_width}" height="540" fill="rgb{MEASURE_RAIL}"/>']
        for lane in range(lanes):
            x = lane_left + lane * 34
            pieces.append(f'<rect x="{x}" y="{top}" width="34" height="540" fill="rgb{LANE_FILLS[lane % 2]}"/>')
            pieces.append(f'<path d="M{x} {top}v540" stroke="#465167"/><text x="{x+17}" y="22" text-anchor="middle">{lane+1}</text>')
        ticks = [Fraction(i, 4) for i in range(math.ceil(base * 4), math.ceil((base + span) * 4))]
        if measure:
            ticks = sorted(set(ticks + [base + i*measure for i in range(COLUMN_MEASURES)]))
        for beat in ticks:
            position = y(beat)
            bar = measure is not None and beat % measure == 0
            whole = beat.denominator == 1
            color = '#c3cddb' if bar else '#394558' if whole else '#2c3748'
            pieces.append(f'<path d="M{lane_left} {position:g}h{lanes*34}" stroke="{color}"/>')
            if bar:
                pieces.append(f'<path d="M{rail_left} {position:g}h{rail_width}" stroke="#c3cddb"/>'
                              f'<text x="{rail_left+rail_width/2:g}" y="{y(beat+measure/2):g}" dominant-baseline="middle" '
                              f'text-anchor="middle">{beat//measure+1}</text>')
            elif whole and measure is None:
                pieces.append(f'<text x="{lane_left-3}" y="{position-2:g}" text-anchor="end">{beat.numerator}</text>')
        pieces.append(f'<path d="M{rail_left} {top}H{lane_right}" stroke="#c3cddb"/>')
        for index, note, start, end in blocks.get(block, ()):
            x = lane_left + 4 + note['lane']*34
            head_y = y(max(start, base))
            tail_y = y(min(end, base+span))
            uncertain = note['status'] == 'unresolved' or bool(note['issues'])
            color = '#ffc857' if uncertain else f'rgb{KEY_COLORS[note["lane"] % 2]}'
            title = escape(f'Lane {note["lane"]+1} · {note["type"]} · Beat {float(start):g} · {note["id"]}')
            pieces.append(f'<a href="#note-{index}" class="chart-note" aria-label="{title}"><title>{title}</title>')
            if note['type'] == 'longnote' and end > start:
                pieces.append(f'<rect x="{x+5}" y="{tail_y:g}" width="16" height="{head_y-tail_y:g}" fill="{color}" fill-opacity=".72"/>')
                if base < end <= base+span:
                    tail_height = min(3, head_y-tail_y)
                    pieces.append(f'<rect x="{x}" y="{tail_y:g}" width="26" height="{tail_height:g}" fill="{color}"/>')
            if base <= start < base+span:
                head_height = min(3, head_y - (tail_y if end > start else top))
                pieces.append(f'<rect x="{x}" y="{head_y-head_height:g}" width="26" height="{head_height:g}" fill="{color}"/>')
                if uncertain:
                    pieces.append(f'<text x="{x+13}" y="{head_y-7:g}" text-anchor="middle" fill="#ffc857">?</text>')
            pieces.append('</a>')
        pieces.append('</svg>')
        panels.append(f'<figure><figcaption>{float(base):g}–{float(base+span):g} beats</figcaption>{"".join(pieces)}</figure>')
    return ('''<style>
.chart-controls{display:flex;gap:1rem;align-items:center;flex-wrap:wrap;margin:1rem 0}
#chart-panels{display:flex;gap:10px;overflow-x:auto;align-items:flex-start;padding:8px;background:#06080b;color:#d0d6de}
#chart-panels figure{flex:0 0 var(--chart-width,350px);margin:0}#chart-panels svg{width:100%;display:block;fill:currentColor}
#chart-panels text{font:11px system-ui}#chart-panels figcaption{text-align:center;padding:8px}
.chart-note:focus rect,.chart-note:hover rect{stroke:#ff668c;stroke-width:3}
tr:target{background:#fff0bf;outline:2px solid #d88a00}tr{scroll-margin-top:2rem}
</style><section aria-label="Chart image"><h2>Chart preview</h2>
<p>Read each column from bottom to top, then move right. Short bars mark taps and connected bars mark holds.
Select a note to see its details below. Yellow notes with a question mark need review.</p>'''
        + ('<p>Barlines follow the measure length in the draft. Measure 1 starts at beat zero.</p>' if measure else
           '<p>The measure length is unknown, so the chart shows beat ticks.</p>')
        + '<p>The chart shows the selected draft notes. The table below includes alternative candidates. '
          'Draft positions are estimates. Empty regions may contain undetected notes.</p>'
        + '<div class="chart-controls"><label>Zoom <input id="chart-zoom" type="range" min="180" max="500" value="350"></label></div>'
        + '<div id="chart-panels">' + (''.join(panels) or '<p>No beat positions to display.</p>') + '</div>'
        + '<h3>Notes outside the chart</h3><ul>' + (''.join(missing) or '<li>None</li>') + '</ul></section>'
        + '<script>document.getElementById("chart-zoom").addEventListener("input",e=>{document.getElementById("chart-panels").style.setProperty("--chart-width",e.target.value+"px");});</script>')
