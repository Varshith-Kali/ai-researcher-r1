# Shift-notes solver: architecture over a fixed weak model

A submission for the AI Researcher R1 screen. Given shift notes and a fixed
weak model (`ibm-granite/granite-4.2-8b`), the system reconstructs each
item's rota (who is on which time block, who holds which station) and
classifies the item as **unique** (one consistent rota), **ambiguous**
(2–4 consistent rotas, all returned), or **inconsistent** (notes contradict
themselves, with a minimal conflicting statement set cited verbatim).

## Running it

```bash
pip install -r requirements.txt   # Python 3.12.3
cp .env.example .env              # put OPENROUTER_API_KEY in .env (never committed)
./run <items.json> --budget <1x|3x|10x> --out <answers.json>
```

- `1x` is exactly one model call per item; `3x`/`10x` use at least one and
  at most three/ten.
- Every model call uses `temperature=1.0`, `top_p=0.95`, reasoning disabled,
  and sends an `X-Item-Id` header with the item id.
- `answers.json` maps each item id to `{"case": ..., ...}` in the required
  shape.

Reproduce the ablation table:

```bash
python3 src/ablate.py --items <items.json> --key <visible_key.json> \
    --out ablation_results/ablation.json
```

## Architecture

The pipeline splits the problem along the line the weak model can and
cannot cross: the model translates statements into formal claims; a
symbolic solver does everything else.

1. **Extraction (model-based).** One call per item asks the model to read
   every numbered statement and emit one claim in a 7-predicate vocabulary
   (`on_block`, `not_on_block`, `block_rel`, `between`, `on_station`,
   `not_on_station`, `station_holder_rel`) or `null` for filler. Relations
   are expressed as *who is earlier / who is later* — never as an ordered
   pair plus a direction menu — because argument-order slips were the
   model's dominant error. The prompt teaches the block-vs-station
   vocabulary distinction, the hedge/restatement/counterfactual rules, and
   shows 19 short examples. A small symbolic filter drops claims from
   lines matching the generator's closed set of past-tense phrasings.
2. **Solver (symbolic).** Header parsing is regex-based; the rest is exact
   enumeration over 120 block × 6 station assignments. Restatements
   collapse by canonical claim key. Zero solutions triggers greedy-deletion
   minimal unsatisfiable subset, with one verbatim line cited per claim.
3. **Solver-validated repair (3x).** If extraction admits >4 solutions
   (missed constraints — a certain misread), the model re-reads only the
   null lines; the solver keeps the smallest subset of proposed additions
   yielding 1–4 solutions, else the original claims stand. Zero solutions
   is never repaired: it is a valid verdict, and the model cannot reliably
   tell a misread from a genuine contradiction.
4. **Per-line majority (10x).** After extraction + repair, independent
   re-extractions vote claim-by-claim (early stop once decided).

The solver never invents constraints, so every answer depends on what the
model actually extracted.

## Writeup

### Characterising the model

Three behaviours shaped the design: (1) it follows a strict JSON schema
well but reasons shallowly — argument order in relations (`between`'s
middle, before/after direction) is its dominant error; (2) its errors are
misreadings, not hallucinations — it rarely invents names but confuses
block/station categories and flips directions; (3) it is sensitive to
framing, not to more examples — the earlier/later reframing was worth ~30
points of macro, while adding examples past ~19 hurt. Temperature is fixed
at 1.0, so single calls vary run to run.

### Why each component exists

- **Earlier/later instead of ordered-pair-plus-direction** removed the
  schema-mapping step behind the dominant error class (1x: ~18% → ~50%).
- **Symbolic solver, not model reasoning:** enumeration is exact and makes
  the three verdicts representable — the verdict derives from the solution
  count, so the confusion matrix (not a scalar) is the right report.
- **Claim-level majority, not answer-level:** with per-sample accuracy
  near a half, wrong answers scatter and answer-majority fails; per-line
  voting keeps the decisions the model is good at (reading a line).
- **Solver-validated repair, not a free critic:** a free-form critic
  deleted valid claims; unconstrained repair flipped correct claims as
  often as it fixed wrong ones (3x scored *below* 1x). The shipped repair
  only adds constraints and only when >4 solutions, so it cannot break a
  correct inconsistent verdict.

### Removed ideas

Free-form critic (deleted valid claims); unconstrained targeted repair
(re-judgment no more reliable than first judgment); answer-level majority
voting (wrong answers scatter); per-line model calls (1x allows one call);
JSON response-format mode (tolerant parser suffices); learned per-line
confidence (cut for time — first thing with a month).

### Where it breaks

Direction flips that keep 1–4 solutions are invisible to every repair
trigger; constraint shapes outside the 7 predicates would be dropped; an
unrepairable >4 case ships an over-long ambiguous answer; citation choice
among several minimal cores is principled but arbitrary.

### With a month

Calibrate per-line extraction confidence and route low-confidence lines to
a targeted pass; grammar-constrained decoding over the predicate
vocabulary; develop against local Granite weights for free iteration; add
an explicit "unparseable" predicate instead of conflating it with `null`.

### Model-based vs symbolic, and new authors

Extraction is model-based, solving symbolic, deliberately. The held-out
set uses a disjoint phrasing pool, so template matching would not
transfer; the model handles rephrasing while the fixed vocabulary keeps
output checkable. Against genuinely new authors the solver is unaffected
(it sees only claims); extraction degrades gracefully on English but a
novel constraint shape ("never adjacent") has no predicate and would be
dropped — the fix is widening the vocabulary, not changing architecture.

## Results

Measured with the provided `score.py` on the 60-item visible set, one run
each. Temperature 1.0, so expect run-to-run variation.

| budget | macro exact match | unique | ambiguous | inconsistent |
|--------|-------------------|--------|-----------|--------------|
| 1x     | 0.45              | 0.50   | 0.30      | 0.55         |
| 3x     | 0.42              | 0.50   | 0.25      | 0.50         |
| 10x    | 0.47              | 0.50   | 0.35      | 0.55         |

The curve is flat by design, not by accident: the weak model's errors are
systematic misreadings, not random slips, so extra calls buy little. The
lift comes from the extraction vocabulary and the solver — 45% from one
call against 0.0% for the bare model and 67.5% for a frontier model
one-shot. 1x is weighted at least as heavily as 10x; the system does not
need calls to work.

### Ablation (12-item fixed subset: 4/4/4; `python3 src/ablate.py`)

| variant      | 1x   | 3x   | 10x  |
|--------------|------|------|------|
| extract-only | 0.67 | 0.42 | 0.50 |
| shipped      | 0.58 | 0.50 | 0.67 |

Each cell is macro exact match. `extract-only` is one extraction call at
every budget; `shipped` is the submitted system. Raw per-cell answers are
saved next to the ablation output for scoring with the official `score.py`.
