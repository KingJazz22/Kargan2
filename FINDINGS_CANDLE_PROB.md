# Probabilistic Candle-State Edge (PCSE) — Findings

> **Update, same session:** the original adaptive-trail exit below was
> disproven, but the permutation test showed the entries themselves carry
> real signal. A systematic exit-design search (see "Update: Exit Strategy
> Search" at the end of this file) found a fixed wide-stop / patient-target
> exit that turns the same entries into a statistically significant winner
> (t=7.11, n=3,821, 25/27 symbols positive). **Read that section for the
> current verdict** — everything below it is the original (superseded)
> exit's post-mortem, kept for the record since it's what motivated the
> search.
>
> **Second update, same session:** the winning config (SL=12x, TP=BB 3.0std)
> was retested on the short side and on 4h/daily timeframes (see "Update 2:
> Short-Side and Cross-Timeframe Retest"). Long confirms robustly on 4h too;
> short fails decisively on every timeframe tested; daily remains too
> sample-starved to trust either way. **Final code**: `strategy_pcse_final.py`
> + `backtest_pcse_final.py`, long-only, hourly or 4h.
>
> **Third update, later session:** a stock-split data bug fixed in
> `alpaca_fetch.py` (unrelated options-spread research surfaced it — see
> `FINDINGS_OPTIONS_SPREADS.md` section 5) meant every result above this
> line was computed on RAW, unadjusted Alpaca bars — a real risk for this
> basket specifically, since it includes five 2016-2026 stock-splitters
> (AAPL, TSLA, NVDA, GOOGL, AMZN). See "Update 3: Re-run on Corrected
> (Split/Dividend-Adjusted) Data" at the end of this file for the full
> re-run. **Bottom line: the verdict is unchanged, and if anything
> slightly stronger** (1h t=7.11→7.18, 4h t=3.87→5.10) — the bug was real
> but turned out not to materially bias this particular model's aggregate
> stats. Safe to keep trading the existing "long-only, 1h or 4h, SL=12x/
> TP=BB(3.0std)" recommendation; no config change needed.

## What this is

An attempt at a strategy that opens trades from candle geometry + statistics +
probability alone (no RSI/MACD/ADX/Bollinger/etc), closes via an adaptive
trailing stop, and is validated with the rigor this project's own history
says is required: walk-forward only (never a single train/test split), full
Alpaca deep multi-regime history (2016–2026 equities, 2021–2026 crypto), long
and short fit and judged completely independently, and a random-entry
permutation test to guard against candle-pattern-mining false discovery.

Code: `indicators_candle.py` (pure OHLCV feature engineering), `model_candle_prob.py`
(walk-forward Markov + logistic blend), `strategy_candle_prob.py` (`prepare()`
entry/exit signal columns), `backtest_candle_prob.py` (event-driven engine +
permutation test), `run_basket_candle_prob.py` / `run_permutation_basket.py`
(CLI runners), `cache_fetch.py` (local disk cache for the deep-history fetches
used only by this strategy's own dev/research loop).

## Design

**Entry** (see `strategy_candle_prob.py`, `model_candle_prob.py`): candle
geometry ratios (body/wick ratios, close-location-value) plus self-referential
rolling statistics (return/range/gap/volume z-scores, rolling percentile
rank) — no canned indicators. A last-3-candle direction sequence (e.g. "UUD")
is treated as a discrete state; each state gets a Beta-Binomial posterior
(reusing this repo's own `bayesian_tracker.BetaBinomialPosterior` from the
grid-martingale strategy) over "does the next 5 bars clear an 8bps forward-
return hurdle," and the entry signal is the posterior's **20th-percentile
credible lower bound**, not its raw mean — a state seen only a handful of
times gets a wide, low bound regardless of how lucky its raw win rate looks,
so sample size is handled by the statistics itself. In parallel, a logistic
regression over the continuous features predicts the same label. Both layers
are refit every 250 bars on a trailing 1500-bar window (walk-forward, never
seeing the fold they're scored on) and must **both** beat their fold's own
unconditional base rate by ≥4 percentage points ("edge margin") to fire —
gating on edge-over-base-rate rather than an absolute probability cutoff,
since an asymmetric threshold label's base rate is rarely near 50%. Long and
short are fit as two independent binary problems throughout.

**Exit**: a two-stage, confidence-adaptive trailing stop. A hard initial stop
at 2.5× the trade's own entry-time return-stdev protects against a fast
reversal; once profit clears 1.0× that stdev the trail engages, sized 1.5–3.0×
stdev by how far the entry confidence exceeded the margin, and ratcheted
tighter at 1R/2R/3R to lock in gains. Independently, if the model's own
edge for the held side decays back below half the entry margin *after* the
model's own prediction horizon has elapsed, the position exits early
("edge_decay").

## Results

### Hourly (primary confirmatory run — 27 symbols, 2016–2026 equities /
### 2021–2026 crypto, long-only)

```
n_trades = 9,158
win_rate = 40.9%
avg_R = -0.149
t_stat_vs_zero = -24.6
profit_factor = 0.51
```

**Every symbol lost money** except ADA-USD (+3.0%, n=7 — too small to mean
anything). 20/20 US equities and 6/7 tradeable crypto symbols were net
negative, from -17% (AMD) to -60% (DOGE-USD, JNJ, WMT). This is not a
borderline result — with n=9,158 and t=-24.6, it is about as statistically
decisive a **negative** result as a backtest produces.

Re-run with both sides independently enabled on SPY: long n=284, avg_R=-0.122,
t=-6.31; **short n=234, avg_R=-0.216, t=-9.45** — short fails independently
and harder than long, consistent with this project's prior (twice-confirmed)
finding that MTF's short side failed independently of its long side. Neither
side has anything resembling an edge on hourly bars.

**Signal-inversion check**: if the model were simply anti-correlated with
outcomes (a sign error, not "no edge"), trading the *opposite* of its signal
should be profitable. On SPY hourly, inverting made it **worse** (long-signal
bars traded short: avg_R=-0.229, t=-11.4, vs -0.122/-6.31 going long as
signaled). Both directions lose against these bars' actual subsequent price
action — ruling out a simple sign bug and pointing instead at "these
candle-shape states just aren't predictive here," which is the more
parsimonious (and expected, under an efficient-markets prior for
easy-to-compute, widely-available pattern features on some of the most
liquid instruments that exist) explanation.

### 4h (spot check, SPY + BTC/USD, long-only)

```
SPY: n=64, avg_R=-0.065, t=-0.87   (not significant, mildly negative)
BTC: n=12, avg_R=-0.409, t=-2.15   (negative, but n too small to trust)
```

Not decisive either way, but not positive.

### Daily (44 pooled trades across the basket, long-only)

```
n_trades = 44, win_rate = 56.8%, avg_R = +0.087, t_stat = 0.77 (not significant alone)
```

Weakly positive but statistically inconclusive on its own (t=0.77 needs to
clear roughly ±2 to be trustworthy). The **random-entry permutation test**
(pooled across the basket, 200 shuffles, entry timing only — same
confidence/volatility columns held fixed) put this result at the **97.5th
percentile** of the null distribution: the model's entry *timing* beat random
timing in 97.5% of trials, which is an intriguing signal on such a small
sample. Both-sides daily was weaker (t=0.47).

**This daily result should NOT be read as "confirmed."** A parameter
sensitivity sweep (`edge_margin` ∈ {0.02,0.03,0.04,0.05} × `horizon` ∈
{3,5,10}, same daily basket) produced t-stats ranging from **-1.25 to +1.87**
with no consistent pattern — the sign and magnitude flip across neighboring,
equally-defensible parameter choices. That fragility, combined with the tiny
sample (daily gives each symbol only ~4 walk-forward folds over the
available history) and the fact that testing 12 parameter cells and reporting
the best one is exactly the kind of multiple-comparisons trap this project's
`FINDINGS_MTF.md` warns about, means the daily "positive" is much more likely
a small-sample fluke than a real edge. The reported daily numbers above use
the **pre-registered** defaults (`EDGE_MARGIN=0.04`, `HORIZON=5`) chosen
before any results were viewed, specifically to avoid cherry-picking from
that sweep.

### Basket-wide permutation test, hourly subset (8 symbols: SPY, QQQ, AAPL,
### JPM, XOM, BTC/USD, ETH/USD, DOGE/USD; 25 shuffles — full 27-symbol × 200-shuffle
### run was computationally impractical in this session, see Limitations)

```
REAL (pooled, long):  n=3,444   avg_R=-0.137   t_stat=-13.66
NULL (random entry, same exit/sizing engine, 25 trials): t_stat ranged approx -14 to -19
null_percentile_long = 100.0
```

This is the single most important — and most nuanced — result in this
report. The real signal's t-stat (-13.66) beat **every one of the 25**
random-entry trials (which averaged closer to t≈-17). In other words: **the
candle-state entries genuinely are better than random timing, consistently
and by a wide margin** — this is real, non-random information content, not
noise. But "better than a random entry into this exact exit/cost engine" is
a much lower bar than "profitable," and the engine's own baseline is
apparently so unfavorable (see the cost-hurdle mismatch and cash-capped
sizing under Filled-in spec gaps) that even a real, consistent entry-timing
edge isn't enough to clear it into positive territory.

## Verdict: **DISPROVEN AS BUILT at the primary (hourly) timeframe — do not
## deploy, but the entry signal itself is not simply noise**

Following this project's own bar (walk-forward + deep multi-regime history +
independent long/short validation, the same standard that correctly flagged
MTF's Yahoo-window result as a bull-market artifact): PCSE **as a complete
trading system is net unprofitable and must not be deployed.** The large-sample
hourly result is decisively negative for both sides independently, and the
one timeframe that looked promising in isolation (daily) is small-sample,
not permutation-airtight at scale, and falls apart under a modest parameter
sweep.

However, this is **not** the same finding as "candle-shape patterns carry no
information" — the permutation test says the opposite: entry timing beat
random timing at the 100th percentile on the tested subset. The signal
inversion check (trading the opposite of the model) also failed, ruling out
a simple sign error. Taken together, the most defensible read is: **the
entries have a real, small, consistently-measurable edge, and the exit/sizing
mechanics around them currently destroy more value than that edge provides**
-- and it's specifically the exit/sizing side, not the cost-hurdle mismatch:
raising `COST_THRESHOLD` to remove that mismatch (tested directly, see Filled-in
spec gaps) made results marginally *worse*, not better, and random entries
into the very same exit engine average an even deeper negative t-stat than
the real signal does. That reframes the highest-priority follow-up (see Ideas
for further work) toward the trailing-stop/ratchet/cash-capped-sizing
mechanics specifically, not the label definition -- the entry signal has
already cleared its own, harder bar (beating a random-timing null, not just
beating zero); it's the machinery around it that needs fixing before
concluding anything further either way.

## Filled-in spec gaps (things the brief left open)

- **Round-trip cost**: label hurdle (`COST_THRESHOLD=0.0008`, 8bps) vs.
  simulated per-trade cost (`cost_per_trade_pct=0.0005` **each way** = 10bps
  round trip) — the label's "success" bar is actually *smaller* than the
  simulated cost of capturing it. **Tested directly** (8-symbol hourly subset,
  `COST_THRESHOLD` raised to 0.0012 and 0.0020): this made results **slightly
  worse**, not better (avg_R -0.137 → -0.149 → -0.155 as the threshold rises),
  so this mismatch is real but is NOT the primary driver of the negative
  result — ruling out the cheapest fix and pointing at the exit/sizing
  mechanics (next bullet) as the more likely culprit, consistent with random
  entries into the same exit engine also averaging a deeply negative t-stat
  (see the permutation test result below).
- **Position sizing hits a real capital constraint on calm hourly bars**:
  `shares = min(risk_amount/stop_distance, cash/entry_price)` — whenever
  `stop_distance/entry_price < risk_pct` (i.e., return-stdev is low relative
  to the 1% risk target, which is routine for hourly SPY/mega-cap bars), the
  cash cap binds and the position ends up sized near 100% of capital instead
  of the intended 1% risk. This isn't a bug so much as what a real cash
  (non-margin) account would face with a tight stop on a large-notional
  asset, but it means the **total-return/equity-curve numbers should be read
  with real risk larger than "1% per trade" on many hourly trades** — the
  pooled R-multiple t-stat is the more trustworthy statistic here since R is
  computed consistently against the nominal risk_amount regardless, and it's
  already the primary verdict-driving number.
- **Credible-interval percentile** (20th) and its **min-sample floor** (20)
  for the Markov layer, and the **3× edge_margin soft cap** used to scale the
  trailing-stop multiplier, are both filled-in gaps with no natural
  "correct" value — chosen for reasonable statistical conservatism, not
  tuned against results.
- **New dedicated Alpaca paper account** for this strategy was scaffolded
  (`.env.pcse.example`) but never provisioned with real credentials or wired
  into a live/paper runner, per this session's backtest-only scope — moot
  anyway given the verdict above.

## Limitations of this validation run

- The basket-wide permutation test at full hourly scale (27 symbols × 200
  shuffles) was not computationally practical in this session — `_simulate`
  re-walks every bar of the test period per trial regardless of trade count,
  making it ~15s/symbol/trial, i.e. ~22 hours for the full grid. A reduced
  8-symbol/25-shuffle version was run instead as a partial confirmation.
  Speeding up `_simulate` (e.g., only iterating bars between candidate
  entries rather than every bar) would make the full test tractable and
  should be the first thing done before trusting a future positive result at
  this scale.
- 4h was only spot-checked on 2 symbols, not run as a full basket.
- The `COST_THRESHOLD` fix was tested on the 8-symbol subset only, not
  re-run across the full 27-symbol basket -- given it made the subset result
  slightly worse, a full re-run is not expected to change the verdict, but
  wasn't done to confirm that at full scale.

## Ideas for further work (not pursued here — out of scope once the primary
## timeframe result was decisive)

- Investigate the exit/sizing mechanics specifically, now that the
  cost-hurdle hypothesis has been ruled out: does loosening the ratchet caps
  (`RATCHET_TRAIL_CAPS`) or widening `HARD_STOP_MULT`/the trail multipliers
  change the *random-entry* baseline's own t-stat? If random entries into a
  looser exit engine come much closer to zero, that confirms the exit
  machinery (not the label) is the main drag, and the already-validated entry
  edge might clear a fixed version into real profitability. A quick,
  single-symbol (SPY), single-random-draw probe of this during this session
  was **inconclusive** -- widening the trail moved both the real and random
  t-stats, but not consistently in the direction the hypothesis predicts, and
  a single random draw on one symbol is far too noisy to trust either way.
  This needs the same pooled-basket, many-shuffle treatment as the main
  permutation test to say anything real, which there wasn't time for here.
- Speed up `_simulate` and run the full permutation grid before trusting any
  future "positive" result at hourly/4h scale — this project has been burned
  before by results that only look good before the right validation is run.
- Test on a less arbitraged universe (small/mid-caps, less liquid altcoins)
  where candle-shape mispricing is more plausible a priori.
- Try longer horizons (multi-day swing) where candle patterns are sometimes
  argued to reflect genuine order-flow/positioning information rather than
  noise, combined with a coarser state space (fewer, better-populated states)
  to reduce the small-sample fragility seen in the daily sweep.

---

## Update: Exit Strategy Search (fixing the exit, not the entries)

### Why: the permutation test said the entries were the wrong thing to blame

The original verdict above found PCSE's entries beat random timing at the
100th percentile, yet the complete system still lost decisively -- the
adaptive trailing stop (tight hard stop at 2.5x stdev, trail engaging at just
1.0x stdev profit, ratcheted tighter at 1R/2R/3R) was the diagnosed culprit:
random entries into that same exit engine did *worse* than the real signal,
not better, meaning the exit machinery destroyed value regardless of entry
quality. This section holds the entries **completely fixed** (same
`strategy_candle_prob.prepare(long_only=True)` walk-forward entry signal, same
symbols, same hourly timeframe) and searches exit design as requested: fixed
TP/SL grids, TP at moving-average touch, and TP at RSI/Stochastic/Bollinger/
Keltner levels.

### Method

`exit_lab.py` freezes each symbol's entry signal once (`prepare_entries()`,
reusing the expensive walk-forward fit) and attaches EMA/rolling-VWAP
(50/100/150/200), RSI(14), Stochastic(14,3,3), Bollinger(20) and Keltner(20,10)
from this repo's own `indicators.py` -- canned indicators are fine for exits;
the brief only required *entries* to come from pure candle statistics. Each
symbol's prepared DataFrame is cached to `.cache_entries_1h/` so every exit
variant below is a cheap ~2s/symbol simulation, not a ~15s walk-forward refit.
SL is always a **fixed** price set at entry (entry_price - sl_mult x
entry-time return-stdev, not trailing); TP is either a fixed price target, a
moving-average/band level (checked intrabar, resting-limit-order convention),
or an indicator threshold (checked on close, filled next open -- same
lookahead-safe convention as entries). Sweeps ran on a 10-symbol subset (SPY,
QQQ, AAPL, MSFT, NVDA, JPM, XOM, BTC-USD, ETH-USD, DOGE-USD) for speed; the
winning configuration was then confirmed on the full 27-symbol basket.

### Phase 1 — fixed TP x SL grid (stdev multiples)

Best of 42 combos tested (SL, TP in {1.0..4.0}x and {1.0..6.0}x): every
top result had the **widest SL tested (4.0x)** and the **widest TP tested
(6.0x)**, with t-stats still only marginally positive (best t=0.34) --
a clear signal that the grid's boundary, not its interior, held the optimum.

### Phase 2 — TP at moving-average touch

At SL=4.0x (the phase-1 winner), TP=EMA200 was the standout: t=2.05, the
only MA-touch rule to clear significance. Shorter MAs got progressively
worse and EMA50/VWAP50 were **negative** (t=-2.18 / -1.54) -- consistent
with the core lesson of this whole search: **short, tight exits keep cutting
this signal's winners off before they develop; patience is the active
ingredient**, not the specific indicator.

| TP | SL | t-stat | avg_R | win% |
|---|---|---|---|---|
| EMA200 | 4.0x | **2.05** | 0.063 | 45.4% |
| VWAP200 | 4.0x | 0.97 | 0.028 | 43.5% |
| VWAP150 | 4.0x | 0.68 | 0.018 | 46.7% |
| EMA150 | 4.0x | 0.61 | 0.016 | 47.3% |
| EMA50 | 4.0x | -2.18 | -0.036 | 57.4% |

### Phase 3 — TP at RSI / Stochastic / Bollinger / Keltner levels

Same pattern, sharper: the *most extreme* (hardest-to-reach) levels of RSI
and Bollinger won decisively; Stochastic never worked at any level tested
(worst: -2.93 at %K>=70); Keltner was mediocre throughout.

| TP | SL | t-stat | avg_R | win% |
|---|---|---|---|---|
| RSI >= 80 | 4.0x | **3.55** | 0.233 | 26.4% |
| BB(3.0 std) | 4.0x | 3.10 | 0.091 | 46.2% |
| RSI >= 75 | 4.0x | 2.34 | 0.086 | 34.9% |
| BB(2.5 std) | 4.0x | 2.25 | 0.044 | 55.3% |
| Stoch >= 70..90 | 4.0x | -2.4 to -2.9 | negative | -- |

Both winners (RSI>=80, BB 3.0std) again hit the phase's SL boundary (4.0x),
so the SL range was extended next.

### Extended SL range (up to 10-20x) against the top TP rules

```
SL=12.0x TP=BB(3.0std)   n=1654  avg_R=0.094  t=5.18  PF=1.38  win%=75.0
SL=20.0x TP=BB(3.0std)   n=1302  avg_R=0.072  t=4.79  PF=1.46  win%=82.9
SL=10.0x TP=BB(3.0std)   n=1775  avg_R=0.093  t=4.62  PF=1.32  win%=70.5
SL=12.0x TP=RSI>=80      n= 857  avg_R=0.203  t=4.61  PF=1.47  win%=54.5
SL=15.0x TP=RSI>=85      n= 421  avg_R=0.427  t=3.43  PF=1.75  win%=44.2
```

Performance is **stable across the entire SL=10x-to-20x neighborhood** (all
give t=4.1-5.2) rather than spiking at one lucky value -- exactly the kind of
broad, boring plateau that's trustworthy, unlike the daily entry-parameter
sweep earlier in this report where the sign flipped between adjacent cells.
**SL=12.0x, TP=BB(3.0 std)** was taken as the winner (peak t-stat, large
enough n).

Combining TP rules ("exit at whichever of these levels is touched first")
was tried and made things **worse** (min(BB3.0, BB2.5): t=3.14; adding EMA200:
t=2.75, both below BB(3.0std) alone at t=4.62-5.18) -- an OR-combination
fires at the *first*, hence *least* patient, of its component levels, which
directly fights the "patience is the edge" finding. Combining wasn't a case
of the best options reinforcing each other; the weaker components dragged
the strongest one down. This is itself a useful, reportable negative result.

### Final confirmation: full 27-symbol basket, hourly, 2016–2026

```
                          n     avg_R   win%    PF   t_stat
SL=12x TP=BB(3.0std)   3,821    0.082  75.5%  1.34    7.11
SL=10x TP=BB(3.0std)   4,110    0.080  71.0%  1.28    6.34
SL=20x TP=RSI>=80       1,593   0.125  63.4%  1.42    5.68
SL=12x TP=RSI>=80       1,912   0.162  52.8%  1.36    5.64
SL=15x TP=RSI>=85         927   0.423  44.7%  1.75    3.71
```

The winner strengthens at full scale (t=5.18 on 10 symbols -> **7.11 on 27**),
which is the opposite of what an overfit result typically does. **25 of 27
symbols were net positive** (only AMD slightly negative at -0.81% total
return / t=-0.10, and TSLA flat at -0.16% / t=0.02); returns ranged up to
+38% (DOGE-USD), +36% (WMT), +32% (AAPL), +31% (V, QQQ). Full per-symbol
table is in `exit_final_1h.log` / reproducible via `run_exit_final.py`.

**Split-sample stability check** (same config, first half vs second half of
each symbol's history, since the exit parameters were chosen by looking at
the *full* period and this isn't a true held-out test):

```
first half (earlier history):  n=2,209  avg_R=0.109  t=7.10  win%=76.6%  PF=1.48
second half (later history):   n=1,606  avg_R=0.045  t=2.61  win%=74.0%  PF=1.18
```

Both halves are independently positive and statistically significant, with
consistent win rates (~75%) -- the edge is real across both eras tested,
though it is **visibly weaker in the more recent half** (t=7.10 -> 2.61,
avg_R roughly halves). That decay is worth taking seriously (see caveats
below) rather than assuming the earlier-half number is what to expect going
forward.

### Revised verdict: **entries + this exit clear the bar — a real, if moderate and possibly-decaying, edge**

Combining this update with the original report: PCSE's candle-state entries
were never the problem (permutation test, 100th percentile). The original
adaptive-trailing-stop exit was the problem, and it has been replaced by a
much simpler rule -- **a wide fixed stop (12x entry-time return-stdev) and a
patient take-profit (price touches the 20-bar Bollinger upper band at 3
standard deviations)** -- that turns the same entries into a statistically
significant, broad-based winner (t=7.11 pooled, 25/27 symbols positive,
stable across a wide SL neighborhood and both halves of a 10-year history).
This is a materially different, much more positive verdict than the original
report's "disproven, do not deploy," and this is the config worth carrying
forward toward paper trading, not the original adaptive-trail design.

**Caveats before deploying, in priority order:**
1. **The exit parameters were selected in-sample** (best-of-N over the full
   period) -- the split-half check confirms the win isn't confined to one
   era, but it is NOT the same rigor as the entries' own walk-forward
   validation. A true nested test (pick exit params on an early slice only,
   confirm on a completely untouched later slice) has not been done.
2. **The edge measurably decayed in the second half of history** (t=7.10 ->
   2.61) -- extrapolate the more recent, weaker number for planning purposes,
   not the full-period or first-half numbers.
3. **A wide 12x-stdev stop plus this project's existing 1%-risk-per-trade
   sizing convention interacts with the same cash-cap constraint** noted in
   the original report -- a wide stop makes the cash cap *less* likely to
   bind (good), but this should be re-verified now that the stop distance
   has changed by ~5x from the original design.
4. ~~Only long was tested here~~ -- **retested, see Update 2 below**: short
   fails independently and decisively with this exit on every timeframe
   tested, confirming this project's long-standing rule never to assume a
   result transfers across sides.
5. ~~This is still hourly-only~~ -- **retested, see Update 2 below**: 4h
   confirms robustly; daily does not have enough bars to say anything either
   way with this entry model's 1500-bar training window.

Reproduce via: `python prep_entries_cache.py 1h full` (cache, one-time) then
`python run_exit_final.py` (final confirmation + split-sample check); the
full parameter sweeps are `run_exit_sweep.py` (phases 1-3) and
`run_exit_sweep2.py` (extended SL range + TP combinations).

---

## Update 2: Short-Side and Cross-Timeframe Retest

Requested follow-up: retest the SL=12x / TP=BB(3.0std) config on the short
side (independently fit and backtested, never assumed symmetric to long) and
on 4h/daily timeframes, and specifically re-examine whether SL=12x still
holds up rather than being a hourly-long-only artifact.

### Method

`backtest_pcse_final.py` was extended to handle short trades as a genuine
mirror image (SL above entry, TP at the Bollinger LOWER band, all cost/cash-
flow directions flipped) and to expose `sl_mult`/`bb_num_std` as function
arguments (not just module constants) so sweeps don't need to monkeypatch
anything. `prep_dual_entries_cache.py` caches entries with `long_only=False`
(both `long_entry` and `short_entry` populated) per timeframe. `run_backtest`
was refactored into `simulate_rows()` + a new `isolate_side()` helper, so
long and short can be backtested completely independently from the SAME
cached entries (one side's entry column forced off) without competing for
the single open-position slot or re-running the walk-forward fit twice. The
refactor was regression-tested against the prior long-only 1h result
(SPY: n=134, avg_R=0.070, t=0.93, exact match before and after).

SL was swept over {4, 6, 8, 10, 12, 15, 20}x for each (side, timeframe)
combination, holding TP=BB(3.0std) fixed (the winner from the original
search) -- 10-symbol subset first, then the full basket for the timeframes
that looked promising.

### Short side: fails decisively on every timeframe

```
[1h]  short, best SL=12.0x:  n=1,704  avg_R=-0.064  t=-3.80   (worst: SL=20x, t=-4.47)
[4h]  short, best SL=20.0x:  n=  506  avg_R=-0.058  t=-2.10   (worst: SL=12x, t=-3.22)
[daily] short: n=13-24, t=-0.36 to -1.30 (all negative, none significant -- too few trades)
```

Every SL value tested, on every timeframe, was negative for short (26-symbol
full basket for 1h/4h). This isn't a borderline call: short should not be
traded with this entry+exit combination, full stop -- consistent with this
project's two prior, independent findings that MTF's short side failed the
same way. `strategy_pcse_final.prepare()` defaults to `long_only=True` for
exactly this reason and should stay that way.

### 4h: confirms robustly at full scale

```
26-symbol full basket, long, TP=BB(3.0std):
  SL=4.0x   n=900  avg_R=0.129  t=2.42
  SL=6.0x   n=776  avg_R=0.139  t=3.19
  SL=8.0x   n=697  avg_R=0.139  t=3.65
  SL=10.0x  n=646  avg_R=0.137  t=4.05
  SL=12.0x  n=608  avg_R=0.120  t=3.87   <- the carried-forward config
  SL=15.0x  n=543  avg_R=0.118  t=4.24
  SL=20.0x  n=478  avg_R=0.124  t=5.15   <- single best t-stat in this grid
```

Every SL from 4x to 20x is positive and significant on 4h, same shape as the
1h result (monotonically improving t-stat as SL widens, plateauing/still
rising at the top of the tested range). SL=12x is solid (t=3.87) but not the
single best point -- SL=20x edges it out (t=5.15), same as it very nearly did
on 1h (SL=12x t=5.18 vs the 1h grid's own SL=20x reading of t=4.79, where 12x
happened to be the peak). Reading both grids together: **the true optimum is
somewhere in the 12x-20x band and is not sharply peaked at exactly 12x** --
12x is a reasonable, non-cherry-picked, cross-timeframe-consistent choice,
but if squeezing out the single best backtested number matters more than
picking one number in advance and leaving it alone, 15-20x tested marginally
better on both timeframes independently. Given this is already an
in-sample-selected parameter (see Update 1's caveat #1), stacking a second
round of in-sample re-optimization on top of it was judged not worth doing
here -- 12x stays the carried-forward default specifically because it was
picked before this retest, not because of it.

**4h split-sample check** (SL=12x, same first-half/second-half method as the
1h check):
```
first half (earlier history):  n=392  avg_R=0.169  t=4.28  win%=76.0%  PF=1.71
second half (later history):   n=215  avg_R=0.035  t=0.71  win%=72.6%  PF=1.13
```
The **same decay pattern as 1h** (t=7.10->2.61 there), independently, on a
completely different bar count and sampling: strong, significant first half,
much weaker (not significant alone, but still positive avg_R and PF>1)
second half. Seeing the identical qualitative pattern show up independently
on two different timeframes is more reassuring than either result alone --
it's less likely to be a coincidence of one specific bar sequence, and more
likely a real (if fading) property of this entry+exit combination. It should
still be read as a warning to size expectations off the more recent, weaker
number, not the full-period or first-half one.

### Daily: still not viable, at any SL, either side

```
23-symbol full basket, long, TP=BB(3.0std): n=10-24 across all SL values,
t ranges -1.19 to +0.57 -- no significant result in either direction.
```

At the 10-symbol subset scale every SL value produced an IDENTICAL n=5 and
t=5.88 for the top results -- a tell that those "different" SL values were
all resolving via the same handful of trades hitting take-profit long before
any of the (very wide) stop distances were ever reached, not genuine
independent evidence. At full 27-symbol scale the picture is honestly just
noisy (sign flips depending on SL, nothing clears significance). This isn't
a new finding so much as a re-confirmation of the original report's
diagnosis: `TRAIN_BARS=1500` consumes the majority of daily's ~2,600-bar
history per symbol, leaving too few walk-forward test folds to say anything
trustworthy. Daily would need either a shorter training window (with its own
re-validation) or a much longer available history to be usable with this
entry model.

### Revised final recommendation

**Trade long-only, on 1h or 4h, with SL=12x / TP=BB(3.0std).** Both
timeframes independently confirm a real, positive, statistically significant
edge (1h: t=7.11 full-period / t=2.61 recent-half; 4h: t=3.87 full-period /
t=0.71 recent-half) with the same honest caveat both times: **size
expectations off the weaker, more recent number, not the flattering
full-period one.** Do not trade short with this configuration on any
timeframe -- it fails decisively and consistently. Do not use daily -- the
entry model's training-window requirement leaves too small a sample to trust
in either direction on that timeframe.

Reproduce via: `python prep_dual_entries_cache.py [1h|4h|daily] full` (cache,
one-time per timeframe) then `python run_sl_retest.py [1h|4h|daily]`.

---

## Update 3: Re-run on Corrected (Split/Dividend-Adjusted) Data

### Why this was needed

Every result above this section was computed from Alpaca stock bars fetched
via `alpaca_fetch.py`. A bug in that module -- fixed in a later session
while doing unrelated options-spread research (see
`FINDINGS_OPTIONS_SPREADS.md` section 5) -- meant every stock fetch used
Alpaca's default RAW (unadjusted) bars instead of split/dividend-adjusted
ones. A stock split shows up in RAW data as a fake single-day price
collapse (e.g. a 4-for-1 split reads as a -75% one-day return), which
corrupts any statistic computed off Close -- directly relevant here, since
PCSE's entries are built entirely from candle geometry ratios and rolling
return/volatility z-scores (`indicators_candle.py`).

This basket (`basket.US_SYMBOLS`) includes five names that split during the
2016-2026 window: **AAPL** (4-for-1, Aug 2020), **TSLA** (5-for-1 Aug 2020,
3-for-1 Aug 2022), **NVDA** (4-for-1 Jul 2021, 10-for-1 Jun 2024),
**GOOGL** (20-for-1, Jul 2022), **AMZN** (20-for-1, Jun 2022). Each of
these would have fed one fake multi-day crash into that symbol's candle
features, rolling z-scores, and forward-return labels, concentrated in
whichever walk-forward fold happened to cover the split date. The fix
(`adjustment=Adjustment.ALL` in `alpaca_fetch.fetch_stock`) now matches
`data.py`'s yfinance fetchers, which always used `auto_adjust=True`.

Fixing the raw bars wasn't enough on its own -- this strategy's own dev
tooling layers two further disk caches on top of the raw OHLCV
(`.cache_entries_1h` / `.cache_dual_entries_{1h,4h,daily}`, built by
`prep_entries_cache.py` / `prep_dual_entries_cache.py`), and both skip
re-fetching if a cached file already exists. Confirmed via file mtimes
that every stock-symbol entry in all of these caches predated the fix by a
day, so all three cache layers had to be cleared for the 20 US stock
symbols (crypto symbols were never affected -- `fetch_crypto` has no
adjustment concept -- and were left untouched) before re-running.

### Method

Cleared the stale stock-symbol cache files, then reproduced this document's
exact confirmation pipeline against freshly-fetched, split/dividend-adjusted
data: `prep_entries_cache.py 1h full` -> `run_exit_final.py` (1h primary
confirmation + split-sample check), then `prep_dual_entries_cache.py 1h
full` / `4h full` -> `run_sl_retest.py 1h` / `4h` (long+short SL sweep,
both timeframes). No code was changed -- same model, same exit rule, same
scripts, only the underlying price data differs.

### Results: essentially unchanged, if anything modestly stronger

**1h primary confirmation** (SL=12x, TP=BB(3.0std)):

| | n | avg_R | win% | PF | t-stat |
|---|---|---|---|---|---|
| Original (corrupted data) | 3,821 | 0.082 | 75.5% | 1.34 | 7.11 |
| Corrected data | 3,863 | 0.082 | 75.7% | 1.34 | **7.18** |

Split-sample check, corrected data: first half n=2,220 avg_R=0.110 t=7.21
win%=77.1% PF=1.49; second half n=1,638 avg_R=0.046 t=2.64 win%=73.8%
PF=1.18 -- matches the original's first/second-half pattern (t=7.10/2.61)
almost exactly, including the same recent-half decay.

**4h confirmation** (SL=12x, TP=BB(3.0std), 26-symbol basket):

| | n | avg_R | t-stat |
|---|---|---|---|
| Original (corrupted data) | 608 | 0.120 | 3.87 |
| Corrected data | 641 | 0.153 | **5.10** |

4h strengthened more than 1h did -- worth reading with a little caution
(more trades and a higher avg_R both moved in the favorable direction on
corrected data, which is what you'd want to see if the fix were pure noise
removal, but it's also the kind of shift that deserves the same "don't
over-trust one lucky re-run" skepticism this document applies everywhere
else; the 1h result, with 6x the sample size, is the steadier evidence
either way).

**Short side** (both timeframes, every SL value tested): still fails
decisively, if anything more so than before.

| | 1h short (SL=12x) | 4h short (best: SL=20x) |
|---|---|---|
| Original (corrupted data) | t=-3.80 | t=-2.10 |
| Corrected data | t=-4.80 | t=-3.15 |

Every SL value on both timeframes was negative in both the original and
corrected runs -- this was never a marginal call, and corrected data makes
it less marginal, not more.

**Split-corrupted symbols individually** (1h, SL=12x/TP=BB3.0std, corrected
data): AAPL n=169 t=+3.20, GOOGL n=82 t=+2.09, AMZN n=133 t=+1.33, NVDA
n=102 t=+1.27, TSLA n=132 t=-0.92. None of these stand out as an outlier
relative to the rest of the 27-symbol basket in either direction -- the
bug was real, but a single corrupted day out of 5,000-13,000 bars per
symbol turned out to be too small a fraction of this model's rolling
windows and walk-forward folds to visibly move that symbol's aggregate
stats. (AMD, previously the only other net-negative symbol in the
original per-symbol table at -0.81%/t=-0.10, is unaffected by this bug --
no split in this window -- and reproduces at essentially the same -0.81%/
t=-0.10 on corrected data, which is a useful sanity check that the
re-fetch pipeline itself introduced no unrelated drift.)

### Verdict: no change to the recommendation

**The corrected-data re-run reconfirms this document's "Revised final
recommendation" without modification: trade long-only, on 1h or 4h, with
SL=12x/TP=BB(3.0std).** The split-adjustment bug was a legitimate,
serious-looking risk when flagged (raw split-day returns are large enough
to distort a model trained on return z-scores) and was worth verifying
rather than assuming away -- but for this specific entry model, the
verification came back negative: no material change to the pooled
statistics, the split-sample decay pattern, or any individual split-
affected symbol's contribution. This is a different outcome from what
might have been feared, and it should not be read as "the caveat never
mattered" -- it mattered enough to require an actual re-run to rule out,
which is what this section did.

One caveat this update does NOT resolve: the exit parameters (SL=12x,
TP=BB3.0std) were originally selected in-sample against the corrupted
data. This re-run confirms the SAME parameters still work well on
corrected data, which is reassuring, but a fully rigorous re-selection
(re-running the original exit-design search from scratch against
corrected data, to check whether SL=12x is still the in-sample optimum
rather than just "still positive") was not done here -- out of scope for a
data-quality verification, worth doing if this strategy's parameters are
ever revisited for other reasons.

Reproduce via: delete the stale entries under `.cache_alpaca`,
`.cache_entries_1h`, and `.cache_dual_entries_{1h,4h}` for any symbol whose
cache predates the `alpaca_fetch.py` fix, then re-run the same commands as
Update 1/Update 2 above.

---

## Update 4: Corrected 2026-08-24 -- gap-through stop-fill bug fixed, final config re-run

### The bug
Both engines behind this document's "final config" verdict --
`backtest_pcse_final.py` (long/short SL, used by `run_sl_retest.py`) and
`exit_lab.py` (used by `run_exit_final.py`, the source of this document's
headline 1h/4h numbers) -- filled the fixed stop-loss at the theoretical
stop price even on bars that gapped straight through it overnight, the same
bug found across ~14 backtest engines in this project (root-cause proof:
`FINDINGS_SWING_STRUCTURE.md`'s correction note). The take-profit leg
(Bollinger-band touch) was already gap-aware in both engines and untouched.
Fixed both independently -- clamp the SL fill to the bar's Open whenever it
already gapped past the stop.

### 1h primary confirmation (`run_exit_final.py`, SL=12x/TP=BB3.0std, 27 symbols)
| | n | avg_R | win% | PF | t-stat |
|---|---|---|---|---|---|
| Before (corrected-data baseline) | 3,863 | 0.082 | 75.7% | 1.34 | 7.18 |
| After (gap-fill fixed) | 3,863 | 0.081 | 75.7% | 1.34 | **7.06** |

Split-sample: first half n=2,220 avg_R=0.109 t=**7.11** (was 7.21), win%=77.1%,
PF=1.48; second half n=1,638 avg_R=0.044 t=**2.57** (was 2.64), win%=73.8%,
PF=1.18 -- both essentially unchanged.

### 1h short side + SL sweep (`run_sl_retest.py 1h`, `backtest_pcse_final.py`)
Long SL=12x reproduces the exit_lab result almost exactly (n=3,863,
avg_R=0.081, t=7.06) -- a useful independent cross-check, since the two
engines were fixed separately and agree to the third decimal. Short side
(SL=12x): t=-4.91 (was -4.80) -- still decisively negative, if anything
slightly more so.

### 4h confirmation (`run_sl_retest.py 4h`, 26-symbol basket)
| | n | avg_R | t-stat |
|---|---|---|---|
| Long SL=12x, before | 641 | 0.153 | 5.10 |
| Long SL=12x, after | 641 | 0.151 | **5.02** |
| Short (best SL), before | -- | -- | -4.80 to -2.10 range |
| Short (best SL), after | -- | -- | -3.15 to -3.91 range |

### Why the effect is so small here
PCSE's carried-forward config uses a very wide fixed stop (12x entry-time
return-stdev) -- gap-through events are rare relative to a tight ATR-based
stop, so the bug barely moved this strategy's numbers, unlike the
pure-trailing-stop strategies elsewhere in this project (Swing-Structure
Breakout, Breakout Hunter, MTF3) where the correction was much larger.

### Original (superseded) adaptive-trail engine, for completeness
`backtest_candle_prob.py` (the ORIGINAL adaptive-trailing-stop exit, already
disproven and superseded by the fixed-SL/TP config above) has the same
trail-stop fill bug, also fixed. Re-ran the primary hourly confirmation
(`run_basket_candle_prob.py 1h long_only`): n=9,278 (was 9,158), avg_R=-0.194
(was -0.149), t=**-35.11** (was -24.6) -- even more decisively negative
post-fix. No change to this document's verdict on that superseded design
("DISPROVEN AS BUILT... do not deploy"); it was never going to become
positive from this fix and didn't.

### Verdict: unchanged. Trade long-only, 1h or 4h, SL=12x/TP=BB(3.0std)
Every number in this document's "final config" recommendation reproduces to
within a few hundredths of a t-stat point post-fix. **This is the strongest
survival of any strategy checked in this project's gap-fill correction
pass** -- PCSE's live/paper-trading validation (`backtest_pcse_final.py`,
wired into `live_trading.py`) stands exactly as it did before this fix.
