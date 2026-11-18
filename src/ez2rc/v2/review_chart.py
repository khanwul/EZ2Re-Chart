"""Self-contained SVG chart view for local draft review."""
from fractions import Fraction
from html import escape
import math

from .chart_layout import (COLUMN_MEASURES, KEY_COLORS, LANE_FILLS, MEASURE_RAIL,
                           chart_columns, fit_hold_columns, number_rail_width)
from .contracts import ratio, read_ratio
from .draft import beat_text, start_choices
from .measure_grid import MeasureGrid


def _chart_notes(notes):
    for index, note in enumerate(notes):
        choices = start_choices(note)
        if choices:
            for start in choices:
                yield index, {**note, 'start_beat': start}
        else:
            yield index, note


def chart_view(draft):
    notes = draft['notes']
    grid = MeasureGrid(draft['projection'])
    lanes = draft['metadata']['lane_count']
    height, top = 600, 34
    blocks = {}
    missing = []
    for index, note in _chart_notes(notes):
        start = read_ratio(note['start_beat']) if note['start_beat'] is not None else None
        end = read_ratio(note['end_beat']) if note['end_beat'] is not None else start
        if start is None:
            missing.append(f'<li><a href="#note-{index}">{escape(note["id"])}</a> · Lane {note["lane"]+1} · Start beat needs review</li>')
            continue
        end = end if end is not None else start
        first = grid.column(start, COLUMN_MEASURES)
        last = grid.column(end, COLUMN_MEASURES, end=True) if end > start else first
        if first < last:
            start, end, first, last = fit_hold_columns(start, end, first, last, lambda block: grid.boundary(block * COLUMN_MEASURES))
        # Bound rendering of malformed or exceptionally long manual notes.
        if last - first > 256:
            missing.append(f'<li><a href="#note-{index}">{escape(note["id"])}</a> · Hold exceeds the display range</li>')
            continue
        for block in range(first, last + 1):
            blocks.setdefault(block, []).append((index, note, start, end))
    rail_width = number_rail_width(blocks, grid if grid.numbered else None)
    rail_left = 4
    lane_left = rail_left + rail_width
    lane_right = lane_left + lanes * 34
    width = lane_right + 4
    panels = []
    for block in chart_columns(blocks):
        first_measure = block * COLUMN_MEASURES
        base = grid.boundary(first_measure)
        span = grid.boundary(first_measure + COLUMN_MEASURES) - base
        scale = 540 / float(span)

        def y(beat):
            return top + float(base + span - beat) * scale

        pieces = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Chart starting at beat {beat_text(ratio(base))}">',
                  f'<rect x="{rail_left}" y="{top}" width="{rail_width}" height="540" fill="rgb{MEASURE_RAIL}"/>']
        for lane in range(lanes):
            x = lane_left + lane * 34
            pieces.append(f'<rect x="{x}" y="{top}" width="34" height="540" fill="rgb{LANE_FILLS[lane % 2]}"/>')
            pieces.append(f'<path d="M{x} {top}v540" stroke="#465167"/><text x="{x+17}" y="22" text-anchor="middle">{lane+1}</text>')
        ticks = [Fraction(i, 4) for i in range(math.ceil(base * 4), math.ceil((base + span) * 4))]
        bars = {grid.boundary(i): i for i in range(first_measure, first_measure + COLUMN_MEASURES)
                if grid.numbered and grid.visible(i)}
        ticks = sorted(set(ticks).union(bars))
        boundary_beats = {base, base + span}
        for beat in ticks:
            position = y(beat)
            bar = beat in bars
            whole = beat.denominator == 1
            color = '#c3cddb' if bar else '#394558' if whole else '#2c3748'
            pieces.append(f'<path d="M{lane_left} {position:g}h{lanes*34}" stroke="{color}"/>')
            if bar:
                boundary_beats.add(beat)
                pieces.append(f'<path d="M{rail_left} {position:g}h{rail_width}" stroke="#c3cddb"/>'
                              f'<text x="{rail_left+rail_width/2:g}" y="{y((beat+grid.boundary(bars[beat]+1))/2):g}" dominant-baseline="middle" '
                              f'text-anchor="middle">{grid.label(bars[beat])}</text>')
            elif whole and not grid.numbered:
                pieces.append(f'<text x="{lane_left-3}" y="{position-2:g}" text-anchor="end">{beat.numerator}</text>')
        if grid.visible(first_measure + COLUMN_MEASURES):
            pieces.append(f'<path d="M{rail_left} {top}H{lane_right}" stroke="#c3cddb"/>')
        for index, note, start, end in blocks.get(block, ()):
            x = lane_left + 4 + note['lane']*34
            head_y = y(max(start, base))
            tail_y = y(min(end, base+span))
            uncertain = note['status'] == 'unresolved' or bool(note['issues'])
            color = '#ffc857' if uncertain else f'rgb{KEY_COLORS[note["lane"] % 2]}'
            title = escape(f'Lane {note["lane"]+1} · {note["type"]} · Beat {beat_text(note["start_beat"])} · {note["id"]}')
            pieces.append(f'<a href="#note-{index}" class="chart-note" aria-label="{title}"><title>{title}</title>')
            if note['type'] == 'longnote' and end > start:
                pieces.append(f'<rect x="{x+5}" y="{tail_y:g}" width="16" height="{head_y-tail_y:g}" fill="{color}" fill-opacity=".72"/>')
                if base < end <= base+span:
                    if end in boundary_beats:
                        pieces.append(f'<path d="M{x} {tail_y:g}h26" stroke="{color}" '
                                      'vector-effect="non-scaling-stroke" shape-rendering="crispEdges"/>')
                    else:
                        tail_height = min(3, head_y-tail_y)
                        pieces.append(f'<rect x="{x}" y="{tail_y:g}" width="26" height="{tail_height:g}" fill="{color}"/>')
            if base <= start < base+span:
                head_height = min(3, head_y - (tail_y if end > start else top))
                pieces.append(f'<rect x="{x}" y="{head_y-head_height:g}" width="26" height="{head_height:g}" fill="{color}"/>')
                if uncertain:
                    pieces.append(f'<text x="{x+13}" y="{head_y-7:g}" text-anchor="middle" fill="#ffc857">?</text>')
            pieces.append('</a>')
        pieces.append('</svg>')
        panels.append(f'<figure><figcaption>{beat_text(ratio(base))}–{beat_text(ratio(base+span))} beats</figcaption>{"".join(pieces)}</figure>')
    return ('''<style>
.chart-controls{display:flex;gap:1rem;align-items:center;flex-wrap:wrap;margin:1rem 0}
#chart-panels{display:flex;gap:10px;overflow-x:auto;align-items:flex-start;padding:8px;background:#06080b;color:#d0d6de}
#chart-panels figure{flex:0 0 var(--chart-width,350px);margin:0}#chart-panels svg{width:100%;display:block;fill:currentColor}
#chart-panels text{font:11px system-ui}#chart-panels figcaption{text-align:center;padding:8px}
.chart-note:focus rect,.chart-note:hover rect{fill:#ff668c}
.chart-note:focus path,.chart-note:hover path{stroke:#ff668c}
tr:target{background:#fff0bf;outline:2px solid #d88a00}tr{scroll-margin-top:2rem}
</style><section aria-label="Chart image"><h2>Chart preview</h2>
<p>Read each column from bottom to top, then move right. Short bars mark taps and connected bars mark holds.
Select a note to see its details below. Yellow notes with a question mark need review.</p>'''
        + ('<p>Measure 1 starts at beat zero. Barlines follow the draft’s measure boundaries.</p>' if grid.numbered else
           '<p>The measure length is unknown, so the chart shows beat ticks.</p>')
        + '<p>Possible starts for the same note link to the same table row. The table also lists alternative notes. '
          'Draft positions are estimates. Empty regions may contain undetected notes.</p>'
        + '<div class="chart-controls"><label>Zoom <input id="chart-zoom" type="range" min="180" max="500" value="350"></label></div>'
        + '<div id="chart-panels">' + (''.join(panels) or '<p>No beat positions to display.</p>') + '</div>'
        + '<h3>Notes outside the chart</h3><ul>' + (''.join(missing) or '<li>None</li>') + '</ul></section>'
        + '<script>document.getElementById("chart-zoom").addEventListener("input",e=>{document.getElementById("chart-panels").style.setProperty("--chart-width",e.target.value+"px");});</script>')
