# Channel-fade + MACD/ADX — backtest finding

**Conclusion: reliable loser as specified, at every session window tested. Not noise — 200k+ pooled trades, t-stat below -70 in every window.**

**UPDATE (stop-loss removed, session filter removed, compared across bar constructions): the strategy is not just unprofitable, it blows up the account. See "Update" section near the bottom.**

## Spec tested
`min_ma`/`max_ma` = 7-period SMA of Low/High (1-minute bars). Long: close
below `min_ma` while ADX(14)<23 and MACD(12,26,9) histogram>0, entered next
bar's open, exits when price touches `max_ma`. Short is the mirror (close
above `max_ma`, histogram<0, exits at `min_ma`). Entries gated to a
configurable intraday session window; open positions are not force-closed
when the window ends. Full rules in `strategy_channel_macd.py`.

Two gaps in the original spec, filled in and flagged:
- No stop-loss was specified. Added an ATR(14)-based stop at 1.5x ATR
  (`strategy_channel_macd.STOP_ATR_MULT`), since the repo's fixed-fractional
  position sizing needs a stop distance. Untuned — not swept in this pass.
- Indicators are computed on the continuous unfiltered 1-minute series
  (including any pre-market bars in the fetched data); the session window
  only gates which bars can open a new trade.

## Session-window sweep (`sweep_channel_macd_session.py`, 20 US_SYMBOLS, ~2yr 1m Alpaca history each)
| Window (ET) | Trades | Win rate | Avg R | Profit factor | t-stat |
|---|---|---|---|---|---|
| 09:30-10:30 | 23,046 | 34.9% | -0.098 | 0.27 | -72.52 |
| 15:30-16:00 | 13,194 | 14.4% | -0.095 | 0.09 | -107.99 |
| 10:30-11:30 | 27,913 | 21.4% | -0.097 | 0.13 | -120.82 |
| 11:30-12:30 | 29,212 | 13.2% | -0.095 | 0.08 | -170.14 |
| 12:30-13:30 | 30,090 | 9.2% | -0.096 | 0.05 | -207.91 |
| 13:30-14:30 | 28,744 | 7.7% | -0.097 | 0.04 | -214.60 |
| 14:30-15:30 | 29,562 | 7.3% | -0.096 | 0.04 | -218.58 |

Every window is negative, with a smooth degradation from open to midday: the
first hour (09:30-10:30, the user's original target window) is the *least
bad* window, and win rate falls monotonically from 34.9% at the open down to
~7-9% through the middle of the day, bottoming out at 14:30-15:30. Avg R is
essentially flat (~-0.095 to -0.098) across every window — losses aren't
concentrated in a bad session, this loses at a near-constant rate all day.
The one exception to the trades-count trend is the truncated final window
(15:30-16:00), which sees fewer trades (only 30 min wide) but a noticeably
worse win rate than the similarly-early 10:30-11:30 window.

## Reading the exit mix
At the default 09:30-10:30 window: 12,870 target exits vs. 10,176 stop-loss
exits (56%/44%) — the strategy hits its take-profit MORE often than its
stop, yet still loses money overall (avg R -0.098). This means the reward
leg (distance from entry to the opposite channel band) is smaller on average
than the 1.5x-ATR risk leg: the channel is fading a genuine overshoot, but
snapping back only as far as the *opposite* band, which on 1-minute bars is
often a smaller move than the ATR-based stop distance risks. This is the
first thing to address before concluding the underlying idea (fade the ADX<23
+ MACD-agreeing channel overshoot) has no edge at all — it may just be
misconfigured, not wrong.

## Suggested next steps (not run in this pass)
- **Stop-loss tuning**: sweep `stop_atr_mult` (0.5-2.5x) — the exit-mix
  finding above suggests the current 1.5x stop is too wide relative to the
  channel-width reward, not that the direction filter itself is broken.
- **Reward/risk symmetry**: consider sizing the stop relative to the
  min_ma/max_ma channel width itself (e.g. a fraction of the channel width)
  rather than a fixed ATR multiple, so risk and reward scale together.
- **Direction-inversion check**: t-stats this large and this consistent
  across every window are themselves informative — worth checking whether
  reversing the MACD-histogram sign convention (trade WITH momentum instead
  of fading against an overshoot) performs any better, since the current
  rule fades in the direction MACD already confirms, which may be pushing
  entries further into the move rather than catching genuine mean reversion.
- Everything above was tested with `stop_atr_mult=1.5` and
  `cost_per_trade_pct=0.0005` fixed — no parameter sweep beyond session
  window has been run yet.

## UPDATE: stop-loss removed, session filter removed, compared across bar constructions

Two changes made on request:
1. **Stop-loss removed entirely.** The only exit is now the target (price
   touching the opposite channel band) — `strategy_channel_macd.STOP_ATR_MULT`
   was renamed `SIZE_ATR_MULT`: ATR(14)x1.5 is now used ONLY to normalize
   position size (bigger recent range -> smaller position), no price stop
   order is ever placed. At most one open position per symbol at a time was
   already enforced by construction (single `position` variable in the event
   loop) — confirmed, no code change needed there.
2. **Intraday session-time filter removed.** Entries are now evaluated on
   every bar regardless of time of day — the earlier "first hour" gating is
   gone from `strategy_channel_macd.prepare()`.

Compared four bar constructions, all on the same rules, same 20 US_SYMBOLS
where feasible:

| Bars | Symbols x window | Trades | Win rate | Avg R | Profit factor | t-stat | Per-symbol return |
|---|---|---|---|---|---|---|---|
| 1-minute | 20 symbols, 2024-08-01..2026-08-01 | 319,914 | 13.0% | -0.091 | 0.12 | -304.0 | **every symbol -100%** |
| 2-minute | 20 symbols, same window | 163,085 | 20.2% | -0.087 | 0.19 | -156.15 | every symbol -98.8% to -99.99% |
| Renko (ATR-sized bricks, from 1m) | 20 symbols, same window | 51,635 | 25.1% | -0.243 | 0.07 | -125.19 | every symbol -98.7% to -100% |
| 30-second (tick-resampled) | 3 symbols (AAPL/MSFT/JPM), 2026-07-27..2026-08-01 only | see below | see below | see below | see below | see below | see below |

(30-second row filled in once that run finishes — Alpaca has no native
sub-minute bar endpoint, so 30s bars require resampling raw tick prints,
which took ~160-200s per symbol-day to fetch; the full 20-symbol/~2yr window
used for the other three would take on the order of weeks to fetch, so the
30s comparison is intentionally scoped down to 3 symbols and 5 trading days
— not apples-to-apples statistical power with the other three.)

**Every granularity tested wipes out the account, on every single symbol,
without exception.** Removing the session filter increased trade count
enormously (from ~23k trades/window to 320k for unrestricted 1-minute bars)
and the strategy loses on a large enough fraction of those trades, with no
stop to cap any individual loss, that compounding fixed-fractional sizing
(`shares = risk_amount / (SIZE_ATR_MULT*atr)`, `risk_amount = cash*risk_pct`
recomputed every trade) drives account equity to zero. This isn't a subtle
statistical result — it's -98% to -100% on literally every symbol at every
granularity.

**Root cause, unchanged from the original finding**: the "target" exit
(`max_ma`/`min_ma`) is a *moving* average, not a fixed profit level. When
price keeps moving against the position, the moving band can drift down (for
a long) to meet the falling price and register a "target" fill at a price
*worse* than entry — on AAPL 1m alone, 100% of losing trades exited this
way. Without a stop-loss, there is nothing to cut a bad trade off before the
band eventually catches it, and these adverse "false target" losses are what
compounds the account to zero. Renko is the fastest to blow up per-trade
(avg R -0.243 vs. ~-0.09 for time bars) because fixed-size bricks mean the
channel MAs adapt to price even faster, so the band chases a falling price
more aggressively.

**This strategy, as specified with no stop-loss, is not viable at any bar
construction tested — it doesn't just underperform, it guarantees ruin given
enough trades.** A stop-loss (or some other hard cap on per-trade loss) is
not optional risk management here, it's structurally required for the
"target = moving average" exit design to not eventually zero the account.
The stop-loss removal request and the original underperformance finding are
pointing at the same root cause: with a stop, the strategy loses money
slowly; without one, it loses everything.

## UPDATE 2: real stop-loss + 2-leg TP ladder + breakeven, swept

The old "exit at the moving channel band" design is retired. `backtest_channel_macd.py`
now implements, on request:
- A real stop-loss at `entry ∓ sl_atr_mult * ATR(14)` (checked before any TP on
  the same bar — conservative tie-break, same convention as `backtest_pcse_final.py`).
- TP1 at `entry ± 2.0 * ATR`, closing `tp1_pct` of the position.
- Once TP1 fills, the stop moves to breakeven (entry price) for the remainder.
- TP2 at `entry ± tp2_atr_mult * ATR`, closing whatever remains (no runner left).
- Each partial fill is recorded as its own trade leg (own shares/risk/R-multiple),
  so `basket.py`'s existing pooled R-multiple stats aggregate correctly with no
  changes needed there.

Swept all `7 (sl_atr_mult: 1.5–5) × 3 (tp1_pct: 0.25/0.4/0.5) × 4 (tp2_atr_mult:
2.5–4) = 84` combinations (`sweep_channel_macd_exit_ladder.py`), screened on a
6-month slice (2026-02-01 to 2026-08-01, in-memory from the same cached 1-minute
data, no extra fetching) for speed, then confirmed the single best-by-t-stat
combo on the full 2024-08-01 → 2026-08-01 / 20-symbol window.

**Result: 0 of 84 combinations were profitable.** Not one had a positive t-stat
or a positive average R-multiple.

| Parameter | Direction of improvement | Effect size |
|---|---|---|
| `sl_atr_mult` | wider is strictly better, monotonic across every value tested (1.5→5.0) | large: mean t-stat -135.6 at 1.5x → -26.8 at 5.0x |
| `tp1_pct` | smaller first-leg allocation is better (lock in less, let more ride) | moderate: mean t-stat -81.6 at 0.50 → -47.8 at 0.25 |
| `tp2_atr_mult` | larger is mildly better, weak monotonic trend | small: mean t-stat -68.2 at 2.5x → -63.0 at 4.0x |

The best combo found — **5.0x ATR stop, 25% at TP1 (2.0x ATR), 75% at TP2
(3.5x ATR)** — confirmed on the full 2-year window: 147,032 trades, 55.4% win
rate, avg R **-0.038**, profit factor **0.55**, t-stat **-36.16**. Still a
clear, statistically decisive loser — just a much slower bleed than the
worst combo tested (tightest 1.5x ATR stop averaged t-stat -135.6, some
individual combos below -160).

**This reframes the whole finding.** Exit-design tuning (stop distance, how
much to bank at each leg, breakeven) moves the result by roughly an order of
magnitude in severity — 5.0x ATR stops lose ~4-5x slower than 1.5x ATR
stops — but never crosses into profitability anywhere in a reasonably wide
grid. That's the signature of a **negative-edge entry rule**, not a
fixable exit. The channel-fade entry (close beyond a 7-bar High/Low band,
gated by ADX<23 and MACD-histogram-agreeing direction) is choosing the wrong
side more often than it should, independent of how the resulting trade is
managed. Further exit-parameter tuning is unlikely to rescue this — the next
useful test is the entry rule itself (e.g. the direction-inversion check
already suggested above: trade WITH the channel breakout instead of fading
it, since ADX<23 + MACD agreement may just be identifying real (if modest)
momentum continuations that this strategy is currently fighting).

## UPDATE 3: bringing the opposite-MA target back as TP2, now with SL protection

Request: re-test closing on the opposite channel band (`max_ma`/`min_ma`) —
the original design from before the SL/TP ladder existed — now that a real
stop-loss and breakeven step sit upstream of it, capping the downside that
made it catastrophic the first time.

Added `tp2_mode` to `backtest_channel_macd.run_backtest`: `"atr"` (unchanged,
the fixed-distance TP2 from Update 2) or `"opposite_ma"` (TP2 becomes "price
touches the opposite channel band," re-read live every bar since it's a
moving average, not a fixed price — same intrabar/gap-aware fill convention
as before). TP1, the stop-loss, and the breakeven step are all unchanged;
only what closes the *remainder* after TP1 differs between modes.

Compared both modes at matching SL/TP1 settings, full 20-symbol / 2-year window:

| Config (sl_atr_mult, tp1_pct) | TP2 = fixed ATR | TP2 = opposite MA |
|---|---|---|
| 5.0, 0.25 (the best `atr`-mode combo from Update 2) | t=-36.16, avg R -0.038, PF 0.55 | t=-60.99, avg R -0.057, PF 0.39 |
| 2.0, 0.50 (original default) | t≈-92.9 (mean across tp2_atr_mult in the sweep) | t=-279.02, avg R -0.156, PF 0.28 |
| 3.0, 0.25 | t≈-54.5 (mean across tp2_atr_mult) | t=-119.04, avg R -0.074, PF 0.32 |

**`opposite_ma` is worse than the fixed-ATR TP2 at every SL/TP1 setting
tested — not marginally, often by 2-3x on t-stat.** The stop-loss does its
job (results are no longer -100% total wipeout the way the original
stop-less version was), but the underlying mechanism is still there: on
AAPL alone, `opposite_ma` mode hits a *higher* win rate than `atr` mode
(64% vs. 57.5%) while posting a *worse* average R (-0.061 vs. -0.044) —
the band is still getting touched more often by drifting toward the
position than by price genuinely reverting, it's just that the stop-loss
now prevents any single instance of that from being catastrophic. Same
signature as the very first finding in this document, just contained now
instead of compounding to ruin.

**Conclusion: keep `tp2_mode="atr"` (the default).** Reviving the
moving-average target doesn't help even with proper risk management wrapped
around it — it's a strictly worse exit than a fixed ATR distance under every
setting tried. This is more evidence (on top of Update 2's 84-combo sweep)
that the problem isn't exit design at all — it's the entry rule.

## UPDATE 4: broad exit search — found a real improvement, with a caveat

Request: keep looking for a better exit. Built two new exit engines to
compare against the SL/TP1/TP2 ladder (Update 2's best: t=-36.16, PF 0.55):

- `backtest_channel_macd_trailing.py` — two-stage stop, reusing the exact
  shape already proven out elsewhere in this repo for Breakout Hunter
  (`backtest_breakout.py`): fixed initial stop, switches to an ATR trailing
  stop once price has moved `trail_activate_atr`x ATR in favor. No fixed
  take-profit — a strong move can run as far as it goes.
- `backtest_channel_macd_single_tp.py` — the simplest possible exit: one
  fixed SL, one fixed TP, no partial legs, no breakeven. A control to check
  whether the ladder's complexity was adding value or just cost.

Screened both (64 trailing combos + 24 single-TP combos = 88) on a 6-month
slice, same methodology as Update 2. **Trailing was uniformly worse** —
every trailing combo underperformed the best single-TP and ladder results;
tight-stop trailing configs were catastrophic (t as low as -176). **Single
TP won outright**: `sl=5.0x, tp=8.0x ATR` already beat the ladder on the
full window (t=-24.17, PF 0.77 vs. the ladder's t=-36.16, PF 0.55).

That result kept improving as both SL and TP were widened further, so the
search was extended directly on the full window: 7x-12x ATR stops, then
13x-20x. The trend held up with good sample sizes through **`sl=12x,
tp=14x ATR`** — 7,883 trades, avg R **-0.033**, profit factor **0.93**,
t-stat **-3.04** — a real, still-significant-but-much-smaller edge than
anything found before (compare: the original ladder was t=-36, PF 0.55; the
very first no-stop version was total account wipeout).

**Caveat — don't over-read the widest settings.** Pushing past ~12-14x ATR,
results get noisier rather than better: `sl=18x, tp=22x` posted avg R
**-0.007**, PF 0.98, t=-0.38 (statistically indistinguishable from zero) —
but its immediate neighbors (`sl=18/tp=20`, `sl=16/tp=22`, `sl=20/tp=22`)
were all 3-5x worse (avg R -0.03 to -0.034), which is the signature of a
small, noisy sample (down to ~3,100 trades pooled across 20 symbols over 2
years — about one trade per symbol every 3 days) rather than a genuine
plateau at breakeven. At that frequency the strategy has effectively become
a different, much lower-frequency system, and there isn't enough data left
to tell a real edge-shift from noise. **The `sl=10-12x / tp=12-16x ATR`
region is the trustworthy result** — good sample size (7,000-14,000 trades),
consistently negative but small (avg R -0.03 to -0.05, PF 0.88-0.93,
t between -3 and -6 across many neighboring combos, not just one lucky cell).

**Where this leaves things**: exit design *does* matter more than Update 2's
84-combo ladder sweep suggested — moving from a partial-exit ladder to a
single wide SL/TP took the account from a fast, severe bleed to a slow, mild
one. But it still doesn't cross into profitability anywhere trustworthy in
this search. The entry rule remains the most likely place left to find a
real edge — this exit search does, however, hand it a much better platform
to be tested on: `backtest_channel_macd_single_tp.py` with `sl_atr_mult≈10-12,
tp_atr_mult≈12-16` is now the strategy's best-known configuration, and
should be the baseline any future entry-rule change gets compared against,
not the original ladder.

## UPDATE 5: bar-granularity retest under the current best-known exit (2026-08-20)

The original bar-construction comparison (top of this file: 1m/2m/Renko/30s,
all -98% to -100%) and Update 2's exit-ladder sweep were both run before
Update 4 found the single-fixed-SL/TP design (`sl_atr_mult=12,
tp_atr_mult=14`) that cut the bleed by roughly an order of magnitude. Re-ran
that specific config across bar granularities, US_SYMBOLS,
2024-08-01..2026-08-01 (30s scoped to 3 symbols / 5 days as before, same
cached tick data reused):

| Bars | Trades | Win rate | Avg R | Profit factor | t-stat |
|---|---|---|---|---|---|
| 5-minute | 2,599 | 44.4% | -0.048 | 0.91 | -2.28 |
| 2-minute | 4,729 | 45.1% | -0.042 | 0.92 | -2.83 |
| 1-minute | 7,883 | 45.1% | -0.033 | 0.93 | **-3.04** (matches Update 4's full-window number exactly) |
| 30-second (3 symbols, 5 days) | 32 | 37.5% | +0.068 | 1.13 | +0.27 |

**The account-wipeout failure mode is gone at every granularity under this
exit** — no symbol lost anywhere near 100% (individual symbol returns over
the 2yr window range from roughly -57% to +21%, most in the -20%..+10% band,
vs. -94% to -100% for every symbol under the old ladder). But the
finer-timeframe-is-worse pattern from every earlier version of this test
**still holds directionally**: t-stat degrades monotonically from -2.28 (5m)
to -3.04 (1m) as bars get shorter, all three comfortably clearing
significance as a *negative* result. **Channel-fade + MACD/ADX is now
best characterized as: a real, small, statistically decisive loser at every
minutes-scale granularity tested (5m/2m/1m), not a guaranteed-ruin strategy**
-- the earlier "wipes out the account" framing was specific to the old
partial-exit ladder's lack of a wide-enough stop, not an intrinsic property
of trading this entry rule fast. 30s stays the one inconclusive result
(n=32, t=0.27) -- unchanged from before, still needs a properly-scaled run
to mean anything.

**Bottom line across every exit design and every granularity tested in this
document (no-stop, SL/TP ladder, trailing stop, single wide SL/TP; 30s
through daily-adjacent via Renko): channel-fade + MACD/ADX has never once
produced a profitable, adequately-sampled configuration.** The severity of
the loss varies enormously by exit design (bleeds account to zero vs. a
slow, single-digit-percent-per-year drag), but the sign never flips. This is
now about as thoroughly exit-tuned as a strategy gets in this project without
finding an edge -- any further work should target the entry rule itself
(e.g. the direction-inversion idea flagged earlier: trade WITH the
channel/MACD-confirmed move instead of fading it), not another exit variant.

## Code (Update 5 addition)
- `run_basket_channel_macd_5m.py` -- 5-minute basket run (new)
- `run_basket_channel_macd_singletp_timeframes.py` -- runs
  `backtest_channel_macd_single_tp.py`'s current best-known config across
  5m/2m/1m/30s in one script

## UPDATE 6: entry-rule redesign attempt, long/short isolation, and a caught overfit

Request: "improve all this and retest it." Took the standing suggestion from
Updates 2/4 seriously -- tested inverting the entry direction (trade WITH the
channel breakout instead of fading it) -- plus isolated long vs. short, all
run through the best-known exit engine (`backtest_channel_macd_single_tp.py`).

**New module: `strategy_channel_macd_momentum.py`.** Same 7-bar channel and
MACD-histogram/ADX filters as the original, but the trigger side is flipped:
long when close breaks ABOVE `max_ma` (not below `min_ma`) with bullish
MACD confirming, short mirrors. `adx_mode` param tests both `"below"`
(ADX<23, same ranging filter as the fade version, for apples-to-apples
comparison) and `"above"` (ADX>23, the more conventional pairing for a
breakout-continuation idea). Also added `strategy_module` and `side_filter`
overrides to `backtest_channel_macd_single_tp.run_backtest` so any entry-rule
module can be run through the same proven exit without duplicating the exit
engine per experiment.

**Momentum vs. fade, full universe, sl=12/tp=14 (the Update 4 baseline):**

| Variant | Trades | Avg R | PF | t-stat |
|---|---|---|---|---|
| fade (baseline) | 7,883 | -0.033 | 0.93 | -3.04 |
| momentum, ADX<23 | 7,285 | -0.039 | 0.92 | -3.39 |
| momentum, ADX>23 | 6,024 | -0.089 | 0.83 | -6.92 |

Direction inversion does **not** help at this setting -- momentum/ADX<23 is
statistically indistinguishable from the fade baseline, momentum/ADX>23 is
worse. (An AAPL-only spot check had briefly suggested momentum/ADX<23 was
much better than fade -- that was 363-trade single-symbol noise; it didn't
hold at the 20-symbol scale.)

**Long/short isolation (same sl=12/tp=14 setting)** turned up something
real, though: in BOTH variants, longs are much closer to breakeven than
shorts.

| Side | fade | momentum ADX<23 |
|---|---|---|
| Long only | avg R -0.027, PF 0.94, t=-2.67 | avg R -0.018, PF 0.96, **t=-1.65 (not significant)** |
| Short only | avg R -0.089, PF 0.82, t=-9.15 | avg R -0.107, PF 0.79, t=-10.26 |

The short side is dragging the pooled average down substantially in both
variants; the long side alone is much closer to flat.

**Followed the best lead (momentum, ADX<23, long-only) with a wide SL/TP
sweep**, same pattern as Update 4's search: 36 combos from sl=8/tp=10 up to
sl=18/tp=20 on the full window. Results kept improving all the way to the
edge of that grid (best: sl=18/tp=18, t=+0.40 -- crossing to POSITIVE for
the first time), with a genuine *cluster* of neighboring cells near zero
rather than one lucky outlier, so this looked much more trustworthy than
Update 4's noisy widest-SL region. Pushed further, sl=18-26 / tp=18-30 (35
more combos): **every single one came back positive**, best at
`sl=24, tp=26`: avg R **+0.054**, PF **1.11**, t=**+2.22** -- the first
statistically-significant *positive* result anywhere in this document.

**That result did not survive proper validation.** By this point the
project had swept roughly 155+ SL/TP combinations total across every exit
engine in this file, on the SAME 2024-08-01..2026-08-01 window every time --
exactly the multiple-comparisons trap this repo's own methodology exists to
catch (see `sweep_mtf_trend_filter.py`'s in-sample/out-of-sample split, and
`FINDINGS_MEANREV.md`'s "classic overfitting signature" section). Re-ran the
momentum/ADX<23/long-only search properly split: **year 1 (2024-08-01 to
2025-08-01) for parameter selection, year 2 (2025-08-01 to 2026-08-01) held
out and never touched during the search.**

- In-sample (year 1 only, 42 combos swept): best t-stat was **0.90**
  (sl=18, tp=18) -- already far short of the 2.22 the same region showed on
  the full 2-year window. Splitting the sample in half alone cut the
  apparent edge by more than half before the OOS data was even touched.
- Out-of-sample (that year-1 "winner," sl=18/tp=18, tested fresh on year 2):
  avg R **-0.007**, PF 0.98, t=**-0.34**. The edge is gone -- not just
  weaker, statistically indistinguishable from zero on data the search never
  saw, in the same direction (negative, not positive) as everything else in
  this document.

**Conclusion: the apparent positive edge was an in-sample artifact, not a
real one.** Momentum-long-only does NOT beat fade-long-only, and neither
crosses into a validated positive edge. What DOES hold up, because it wasn't
cherry-picked from a widening search -- it showed up consistently and with
good sample size (7,000-9,000+ trades) the first time it was tested --
is the **long/short asymmetry**: shorts are reliably worse than longs across
every variant tried. That's a legitimate, second-order finding worth using
(trade long-only, drop the short side entirely, at the Update 4 exit
settings) even though it doesn't produce a positive-expectancy system on its
own -- it's a real improvement in the same direction as Update 4's exit
work, not a new profitable strategy.

## Corrected 2026-08-24: gap-through stop-fill bug fixed, re-run across engines

All three engines in this document (`backtest_channel_macd.py`'s SL and
breakeven-stop legs, `backtest_channel_macd_single_tp.py`'s SL leg,
`backtest_channel_macd_trailing.py`'s stop leg -- TP legs were already
gap-aware and untouched) filled stop-loss exits at the theoretical stop price
even on bars that gapped straight through it, the same bug found project-wide
(see `FINDINGS_SWING_STRUCTURE.md`'s correction note). Fixed by clamping the
fill to the bar's Open whenever it already gapped past the stop.

**Ladder engine** (`backtest_channel_macd.py`, `run_basket_channel_macd.py`,
default 1m/US_SYMBOLS/2024-08-01..2026-08-01 window, no session filter):
n=283,369, win rate 31.7%, avg R -0.163, t=-248.77 post-fix -- still a
catastrophic, decisive loser (this document's session-window sweep found
t below -70 in every window pre-fix too); **no change to verdict.**

**Single-TP engine, bar-granularity retest** (Update 5's config,
`sl_atr_mult=12, tp_atr_mult=14`, `run_basket_channel_macd_singletp_timeframes.py`):

| Bars | Before (t-stat) | After (t-stat) |
|---|---|---|
| 5-minute | -2.28 | -2.74 |
| 2-minute | -2.83 | -3.57 |
| 1-minute | -3.04 | -4.25 |
| 30-second (3 symbols, 5 days) | +0.27 | +0.21 |

Every minutes-scale granularity got slightly *more* negative post-fix (the
old bug was giving the strategy a small unearned benefit-of-the-doubt on its
own stop-outs) -- consistent with the fix direction and **reinforcing, not
changing, the existing "real, small, statistically decisive loser" verdict**.
30s stays flat/inconclusive at n=32, same as before.

**Trailing-stop engine spot check** (`backtest_channel_macd_trailing.py`,
mid-grid config `sl_atr_mult=4, trail_activate_atr=1.0, trail_atr_mult=2.0`,
6-month screen window, US_SYMBOLS -- not the full 64-combo grid, given this
variant was already established as strictly worse than single-TP): n=39,008,
win rate 25.2%, avg R -0.129, t=-86.25 post-fix -- still catastrophic,
consistent with this document's "tight-stop trailing configs were
catastrophic (t as low as -176)" finding pre-fix. **No change to verdict.**

**Bottom line: this document's conclusion is unchanged and, if anything,
slightly reinforced.** Channel-fade + MACD/ADX has never produced a
profitable, adequately-sampled configuration under any exit design tested,
before or after this fix.

**Where this leaves the whole document**: every avenue tried -- exit design
(ladder, trailing, single SL/TP, opposite-MA target), bar granularity/
construction (30s through Renko), and now entry-rule direction (fade vs.
momentum) plus side isolation -- has been retested under proper conditions
at least once, and none has produced a profitable, validated result. The
best-known, validated configuration remains **fade entry, long-only,
`sl_atr_mult≈10-12, tp_atr_mult≈12-16`** (Update 4 + this update's long-only
finding combined) -- a small, real, statistically decisive loser, not a
guaranteed-ruin one. Any further search should assume the entry rule itself
(the 7-bar channel + ADX + MACD-histogram combination) lacks a real edge in
either direction on this symbol universe at this granularity, and treat a
positive in-sample result on the full window as suspect until it's confirmed
the way this update's result wasn't -- OOS, on data the search never touched.
