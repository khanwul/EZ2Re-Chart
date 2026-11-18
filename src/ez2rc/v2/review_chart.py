"""Self-contained SVG chart view for local draft review."""
from fractions import Fraction
from html import escape
import math

from .contracts import read_ratio


def chart_view(draft):
    notes = draft['notes']
    measure = draft['projection'].get('measure_beats')
    measure = read_ratio(measure) if measure is not None else None
    if measure is not None and measure <= 0:
        measure = None
    span = measure * 4 if measure else Fraction(16)
    lanes = draft['metadata']['lane_count']
    width, height, top, scale = 64 + lanes * 34, 600, 34, 540 / float(span)
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
    panels = []
    for block, entries in sorted(blocks.items()):
        base = block * span
        pieces = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Chart starting at beat {float(base):g}">',
                  f'<rect x="52" y="{top}" width="{lanes*34}" height="540" fill="#17202e"/>']
        for lane in range(lanes):
            x = 52 + lane * 34
            pieces.append(f'<path d="M{x} {top}v540" stroke="#465167"/><text x="{x+17}" y="22" text-anchor="middle">{lane+1}</text>')
        ticks = [Fraction(i, 4) for i in range(math.ceil(base * 4), math.ceil((base + span) * 4))]
        if measure:
            ticks = sorted(set(ticks + [base + i*measure for i in range(4)]))
        for beat in ticks:
            y = top + float(beat-base)*scale
            bar = measure is not None and beat % measure == 0
            whole = beat.denominator == 1
            color = '#c3cddb' if bar else '#394558' if whole else '#2c3748'
            pieces.append(f'<path d="M52 {y:g}h{lanes*34}" stroke="{color}"/>')
            if bar or whole:
                label = f'M{int(beat/measure)+1}' if bar else f'{float(beat):g}'
                pieces.append(f'<text x="47" y="{y+4:g}" text-anchor="end">{label}</text>')
        for index, note, start, end in entries:
            x = 56 + note['lane']*34
            y = top + float(max(start, base)-base)*scale
            bottom = top + float(min(end, base+span)-base)*scale
            uncertain = note['status'] == 'unresolved' or bool(note['issues'])
            color = '#ffc857' if uncertain else '#64d8ef' if note['lane']%2 else '#eff5ff'
            title = escape(f'Lane {note["lane"]+1} · {note["type"]} · Beat {float(start):g} · {note["id"]}')
            pieces.append(f'<a href="#note-{index}" class="chart-note" aria-label="{title}"><title>{title}</title>')
            if note['type'] == 'longnote' and end > start:
                pieces.append(f'<rect x="{x+6}" y="{y:g}" width="14" height="{bottom-y:g}" fill="{color}" fill-opacity=".5"/>')
                if base < end <= base+span:
                    tail_height = min(4, bottom-y)
                    pieces.append(f'<rect x="{x}" y="{bottom-tail_height:g}" width="26" height="{tail_height:g}" fill="{color}"/>')
            if base <= start < base+span:
                head_height = min(6, bottom-y) if end > start else 6
                pieces.append(f'<rect x="{x}" y="{y:g}" width="26" height="{head_height:g}" rx="1" fill="{color}"/>')
                if uncertain:
                    pieces.append(f'<text x="{x+13}" y="{y-7:g}" text-anchor="middle" fill="#ffc857">?</text>')
            pieces.append('</a>')
        pieces.append('</svg>')
        panels.append(f'<figure><figcaption>{float(base):g}–{float(base+span):g} beats</figcaption>{"".join(pieces)}</figure>')
    return ('''<style>
.chart-controls{display:flex;gap:1rem;align-items:center;flex-wrap:wrap;margin:1rem 0}
#chart-panels{display:flex;gap:16px;overflow-x:auto;align-items:flex-start;padding:8px;background:#e9edf3}
#chart-panels figure{flex:0 0 var(--chart-width,250px);margin:0}#chart-panels svg{width:100%;display:block;background:#f9fbff}
#chart-panels text{font:11px system-ui}#chart-panels figcaption{text-align:center;padding:8px}
.chart-note:focus rect,.chart-note:hover rect{stroke:#ff668c;stroke-width:3}
tr:target{background:#fff0bf;outline:2px solid #d88a00}tr{scroll-margin-top:2rem}
</style><section aria-label="Chart image"><h2>Chart preview</h2>
<p>Read each column from top to bottom, then move right. Short bars mark taps and connected bars mark holds.
Select a note to see its details below. Yellow notes with a question mark need review.</p>'''
        + ('<p>Barlines follow the measure length in the draft. M1 marks the reference measure.</p>' if measure else
           '<p>The measure length is unknown, so the chart shows beat ticks.</p>')
        + '<p>The chart shows the selected draft notes. The table below includes alternative candidates. '
          'Draft positions are estimates. Empty regions may contain undetected notes. Empty columns are omitted.</p>'
        + '<div class="chart-controls"><label>Zoom <input id="chart-zoom" type="range" min="180" max="500" value="250"></label></div>'
        + '<div id="chart-panels">' + (''.join(panels) or '<p>No beat positions to display.</p>') + '</div>'
        + '<h3>Notes outside the chart</h3><ul>' + (''.join(missing) or '<li>None</li>') + '</ul></section>'
        + '<script>document.getElementById("chart-zoom").addEventListener("input",e=>{document.getElementById("chart-panels").style.setProperty("--chart-width",e.target.value+"px");});</script>')
