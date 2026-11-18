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
            missing.append(f'<li><a href="#note-{index}">{escape(note["id"])}</a> · {note["lane"]+1}레인 · 시작 박자 확인 필요</li>')
            continue
        end = end if end is not None else start
        first, last = math.floor(start / span), math.floor(end / span)
        # Bound rendering of malformed or exceptionally long manual notes.
        if last - first > 256:
            missing.append(f'<li><a href="#note-{index}">{escape(note["id"])}</a> · 표시 범위를 초과한 롱노트</li>')
            continue
        for block in range(first, last + 1):
            blocks.setdefault(block, []).append((index, note, start, end))
    panels = []
    for block, entries in sorted(blocks.items()):
        base = block * span
        pieces = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{float(base):g}박부터 채보">',
                  f'<rect x="52" y="{top}" width="{lanes*34}" height="540" fill="#17202e"/>']
        for lane in range(lanes):
            x = 52 + lane * 34
            pieces.append(f'<path d="M{x} {top}v540" stroke="#465167"/><text x="{x+17}" y="22" text-anchor="middle">{lane+1}</text>')
        ticks = [Fraction(i) for i in range(math.ceil(base), math.ceil(base+span))]
        if measure:
            ticks = sorted(set(ticks + [base + i*measure for i in range(4)]))
        for beat in ticks:
            y = top + float(beat-base)*scale
            bar = measure is not None and beat % measure == 0
            label = f'M{int(beat/measure)+1}' if bar else f'{float(beat):g}'
            pieces.append(f'<path d="M52 {y:g}h{lanes*34}" stroke="{ "#c3cddb" if bar else "#394558"}"/>'
                          f'<text x="47" y="{y+4:g}" text-anchor="end">{label}</text>')
        for index, note, start, end in entries:
            x = 56 + note['lane']*34
            y = top + float(max(start, base)-base)*scale
            bottom = top + float(min(end, base+span)-base)*scale
            uncertain = note['status'] == 'unresolved' or bool(note['issues'])
            color = '#ffc857' if uncertain else '#64d8ef' if note['lane']%2 else '#eff5ff'
            title = escape(f'{note["lane"]+1}레인 · {note["type"]} · {float(start):g}박 · {note["id"]}')
            pieces.append(f'<a href="#note-{index}" class="chart-note" aria-label="{title}"><title>{title}</title>')
            if note['type'] == 'longnote' and end > start:
                pieces.append(f'<rect x="{x+6}" y="{y:g}" width="14" height="{max(2,bottom-y):g}" fill="{color}" fill-opacity=".5"/>')
                if base <= end < base+span:
                    pieces.append(f'<rect x="{x}" y="{bottom-2:g}" width="26" height="4" fill="{color}"/>')
            if base <= start < base+span:
                pieces.append(f'<rect x="{x}" y="{y-3:g}" width="26" height="6" rx="1" fill="{color}"/>')
                if uncertain:
                    pieces.append(f'<text x="{x+13}" y="{y-7:g}" text-anchor="middle" fill="#ffc857">?</text>')
            pieces.append('</a>')
        pieces.append('</svg>')
        panels.append(f'<figure><figcaption>{float(base):g}–{float(base+span):g}박</figcaption>{"".join(pieces)}</figure>')
    return ('''<style>
.chart-controls{display:flex;gap:1rem;align-items:center;flex-wrap:wrap;margin:1rem 0}
#chart-panels{display:flex;gap:16px;overflow-x:auto;align-items:flex-start;padding:8px;background:#e9edf3}
#chart-panels figure{flex:0 0 var(--chart-width,250px);margin:0}#chart-panels svg{width:100%;display:block;background:#f9fbff}
#chart-panels text{font:11px system-ui}#chart-panels figcaption{text-align:center;padding:8px}
.chart-note:focus rect,.chart-note:hover rect{stroke:#ff668c;stroke-width:3}
tr:target{background:#fff0bf;outline:2px solid #d88a00}tr{scroll-margin-top:2rem}
</style><section aria-label="채보 그림"><h2>채보 미리보기</h2>
<p>위에서 아래로 읽고 다음 열로 이동합니다. 막대는 일반 노트, 연결된 막대는 롱노트입니다.
노트를 누르면 아래 상세 표로 이동합니다. 노란색·? 표시는 확인이 필요한 후보입니다.</p>'''
        + ('<p>마디선은 초안의 마디 길이를 따릅니다. M1은 기준 마디입니다.</p>' if measure else
           '<p>이 초안에는 마디 길이가 없어 마디 번호 대신 박자 눈금을 표시합니다.</p>')
        + '<p>채택된 초안 노트만 그립니다. 대안 후보는 아래 상세 표에서 확인하세요. '
          '초안 위치는 추정값입니다. 빈 구간에도 미검출 노트가 있을 수 있습니다. 노트가 없는 열은 생략합니다.</p>'
        + '<div class="chart-controls"><label>확대 <input id="chart-zoom" type="range" min="180" max="500" value="250"></label></div>'
        + '<div id="chart-panels">' + (''.join(panels) or '<p>표시 가능한 박자 위치가 없습니다.</p>') + '</div>'
        + '<h3>그림에 배치하지 못한 노트</h3><ul>' + (''.join(missing) or '<li>없음</li>') + '</ul></section>'
        + '<script>document.getElementById("chart-zoom").addEventListener("input",e=>{document.getElementById("chart-panels").style.setProperty("--chart-width",e.target.value+"px");});</script>')
