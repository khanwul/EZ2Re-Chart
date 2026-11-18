"""Column order and RGB colors used by both draft chart renderers."""

COLUMN_MEASURES = 4
COLUMN_HEIGHT = 960
LANE_FILLS = ((13, 15, 19), (23, 26, 32))
KEY_COLORS = ((232, 238, 247), (42, 145, 255))
MEASURE_RAIL = (78, 83, 92)


def fit_hold_columns(start, end, first, last, column_start):
    """Put a segment that rounds to zero rows at a seam with the rest of its hold."""
    if first < last:
        edge = column_start(first + 1)
        if 2 * COLUMN_HEIGHT * (edge - start) <= edge - column_start(first):
            start, first = edge, first + 1
    if first < last:
        edge = column_start(last)
        if 2 * COLUMN_HEIGHT * (end - edge) <= column_start(last + 1) - edge:
            end, last = edge, last - 1
    return start, end, first, last


def number_rail_width(blocks, measure):
    """Fit the widest signed measure or beat number and keep columns aligned."""
    per_column, origin = (COLUMN_MEASURES, 1) if measure is not None else (16, 0)
    first = origin + min(blocks, default=0) * per_column
    last = origin + (max(blocks, default=0) + 1) * per_column - 1
    return 8 * max(len(str(first)), len(str(last))) + 7


def chart_columns(blocks):
    """Include rests from beat zero, skipping exceptionally distant manual positions."""
    previous = min(0, min(blocks, default=0)) - 1
    for block in sorted(blocks):
        if block - previous <= 256:
            yield from range(previous + 1, block + 1)
        else:
            yield block
        previous = block
