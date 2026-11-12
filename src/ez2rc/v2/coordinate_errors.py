"""Shared coordinate errors without independence or Gaussian assumptions.

An affine coordinate is a nominal value plus named error sources. Reusing a
source means reusing the same unknown value, including when it is unbounded.
Distinct sources have unspecified dependence; their box is an outer enclosure.
"""
from dataclasses import dataclass
from fractions import Fraction

from .contracts import exact_ms


@dataclass(frozen=True)
class Coordinate:
    nominal: Fraction
    terms: dict[str, Fraction]

    def __post_init__(self):
        object.__setattr__(self, 'nominal', exact_ms(self.nominal))
        object.__setattr__(self, 'terms', {k: exact_ms(v) for k, v in self.terms.items() if v != 0})

    def __sub__(self, other):
        keys = self.terms.keys() | other.terms.keys()
        terms = {key: self.terms.get(key, Fraction(0)) - other.terms.get(key, Fraction(0))
                 for key in keys}
        return Coordinate(self.nominal - other.nominal, {k: v for k, v in terms.items() if v})

    def domain(self, sources):
        lower = upper = self.nominal
        for key, coefficient in self.terms.items():
            if not coefficient:
                continue
            if sources[key] is None:
                return None
            a, b = sources[key]
            lo, hi = sorted((coefficient * a, coefficient * b))
            lower += lo
            upper += hi
        return lower, upper


def coordinate(nominal, domain, terms, sources, residual_id):
    """Separate supplied residual bounds from explicitly named shared errors."""
    nominal = exact_ms(nominal)
    lower, upper = map(exact_ms, domain)
    if not lower <= nominal <= upper:
        raise ValueError('coordinate nominal must lie in its residual bounds')
    result = {key: exact_ms(value) for key, value in terms.items() if value != 0}
    if lower != upper:
        if residual_id in sources:
            raise ValueError('coordinate residual identity collision')
        sources[residual_id] = (lower - nominal, upper - nominal)
        result[residual_id] = Fraction(1)
    return Coordinate(nominal, result)


def quotient_range(numerator, denominator, sources):
    """Exact extrema of an affine quotient over a box with positive divisor.

Each iteration minimizes/maximizes N - r D at a box vertex. With D positive,
the new ratio improves monotonically until the exact rational residual is zero.
Common terms have already cancelled by symbolic subtraction, before bounding.
"""
    n_domain, d_domain = numerator.domain(sources), denominator.domain(sources)
    if n_domain is None or d_domain is None:
        return None, 'unbounded_relative_coordinate_error'
    if d_domain[0] <= 0:
        return None, 'anchor_order_not_established'
    keys = sorted(numerator.terms.keys() | denominator.terms.keys())

    def extreme(maximize):
        value = Fraction(0)
        while True:
            n, d = numerator.nominal, denominator.nominal
            for key in keys:
                nc = numerator.terms.get(key, Fraction(0))
                dc = denominator.terms.get(key, Fraction(0))
                slope = nc - value * dc
                lower, upper = sources[key]
                x = upper if (slope > 0 if maximize else slope < 0) else lower
                n += nc * x
                d += dc * x
            next_value = n / d
            if next_value == value:
                return value
            value = next_value

    return (extreme(False), extreme(True)), None
