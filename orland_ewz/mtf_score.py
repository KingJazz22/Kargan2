"""Combine per-timeframe regime classifications via combinatorial alignment.

Instead of a single blended score, this searches combinations of COMBO_SIZE
timeframes (out of however many are provided -- 6 live, 3 in the deep
backtest) for one subset that's unanimously trending the same direction.
That's what "MTF-confirmed" means. No single timeframe can force or veto a
trade on its own, but unlike a plain weighted score, a clean alignment is
required to actually exist somewhere in the set.

Confirmation requires just 5m and 15m to be trending the same direction --
no other timeframe is needed.

Ported byte-identical from OrlandMagics/mtf_score.py (pure logic, no
pandas/indicators dependency, so nothing needed changing).
"""
from dataclasses import dataclass
from itertools import combinations

COMBO_SIZE = 2
MANDATORY_TIMEFRAMES = {"5m", "15m"}  # the only two timeframes that matter
STRONG_TAG_FRACTION = 0.75  # fraction of the matched combo that must be 'strong' tier for the STRONG alert tag

_TIER_WEIGHT = {"strong": 1.0, "medium": 0.5, "consolidation": 0.0}


def _direction_tier(regime: str):
    if regime == "consolidation":
        return None, None
    direction, tier = regime.replace("trending_", "").rsplit("_", 1)
    return direction, tier


def _vote(regime: str) -> float:
    direction, tier = _direction_tier(regime)
    if direction is None:
        return 0.0
    weight = _TIER_WEIGHT[tier]
    return weight if direction == "up" else -weight


@dataclass
class MTFResult:
    score: float
    direction: str  # "up" / "down" / "none"
    confirmed: bool
    intensity: str  # "STRONG" / "MEDIUM" / "NONE"
    per_timeframe: dict
    matched_combo: tuple = ()


def _find_confirmed_combo(regimes: dict, mandatory: set, combo_size: int):
    timeframes = list(regimes.keys())
    k = min(combo_size, len(timeframes))
    if k == 0:
        return None, None
    for combo in combinations(timeframes, k):
        if mandatory and not mandatory.issubset(combo):
            continue
        directions = set()
        ok = True
        for tf in combo:
            direction, _tier = _direction_tier(regimes[tf])
            if direction is None:
                ok = False
                break
            directions.add(direction)
        if ok and len(directions) == 1:
            return combo, directions.pop()
    return None, None


def compute_mtf_score(regimes: dict, symbol: str = None, combo_size: int = COMBO_SIZE,
                       require_mandatory: bool = True) -> MTFResult:
    """`regimes` maps timeframe name -> regime string (from regime_filter).
    `symbol` is accepted for call-site stability but no longer changes the
    mandatory set -- 5m+15m are required for every symbol now. Set
    `require_mandatory=False` for contexts that only ever see a timeframe
    subset without 5m/15m (e.g. the deep daily-only backtest's exit-rule
    management proxy) -- otherwise confirmation would be permanently
    impossible there."""
    mandatory = MANDATORY_TIMEFRAMES if require_mandatory else set()

    votes = {tf: _vote(r) for tf, r in regimes.items()}
    score = sum(votes.values())

    if mandatory and not mandatory.issubset(regimes.keys()):
        combo, direction = None, None
    else:
        combo, direction = _find_confirmed_combo(regimes, mandatory, combo_size)

    if combo is None:
        return MTFResult(score=score, direction="none", confirmed=False,
                          intensity="NONE", per_timeframe=regimes)

    strong_count = sum(1 for tf in combo if _direction_tier(regimes[tf])[1] == "strong")
    intensity = "STRONG" if strong_count / len(combo) >= STRONG_TAG_FRACTION else "MEDIUM"

    return MTFResult(score=score, direction=direction, confirmed=True, intensity=intensity,
                      per_timeframe=regimes, matched_combo=combo)
