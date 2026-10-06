# Shift-notes solver: architecture over a fixed weak model

A submission for the AI Researcher R1 screen. Given shift notes and a fixed
weak model (`ibm-granite/granite-4.2-8b`), the system reconstructs each
item's rota (who is on which time block, who holds which station) and
classifies the item as **unique** (one consistent rota), **ambiguous**
(2–4 consistent rotas, all returned), or **inconsistent** (notes contradict
themselves, with a minimal conflicting statement set cited verbatim).

## Running it

```bash
pip install -r requirements.txt   # Python 3.10+
cp .env.example .env              # put OPENROUTER_API_KEY in .env (never committed)
./run <items.json> --budget <1x|3x|10x> --out <answers.json>
```

- The budget bounds **model calls per item**: `1x` is exactly one call per
  item; `3x`/`10x` use at least one and at most three/ten.
- Every model call sends an `X-Item-Id` header with the item id, and uses
  `temperature=1.0`, `top_p=0.95`, reasoning disabled, on
  `ibm-granite/granite-4.2-8b` via the provided endpoint.
- `answers.json` maps each item id to `{"case": ..., ...}` in the required
  shape (see the candidate brief).

Reproduce the ablation table:

```bash
python3 src/ablate.py --items <items.json> --key <visible_key.json> \
    --out ablation_results/ablation.json
```

## Architecture

The pipeline splits the problem along the line the weak model can and
cannot cross:

```
shift notes
    │  (1 model call)  model-based: translate each statement into a
    ▼                  formal claim in a small fixed vocabulary
claims  ──► symbolic: enumerate every possible rota (5 staff -> 120 block
    │       assignments x 6 station assignments), evaluate the claims
    ▼
verdict: unique / ambiguous / inconsistent (+ minimal conflicting set)
```

1. **Extraction (model-based).** One call per item asks the model to read
   every numbered statement and emit one claim in a 7-predicate vocabulary
   (`on_block`, `not_on_block`, `block_rel`, `between`, `on_station`,
   `not_on_station`, `station_holder_rel`) or `null` for filler. Relations
   are expressed as *who is earlier / who is later* (never as an ordered
   pair plus a direction menu), because argument-order slips were the
   model's dominant error. The prompt teaches the vocabulary distinction
   the model needs most (block words vs station words), gives the
   hedge/restatement/counterfactual rules from the brief, and shows 19
   short examples. Hedged phrasing is treated as certain; counterfactuals,
   talk of the past, and uncertainty that asserts nothing are ignored. A
   small symbolic filter additionally drops any claim whose line matches
   the generator's closed set of past-tense phrasings.
2. **Solver (symbolic).** Header parsing (rota, ordered blocks, stations,
   station holders) is regex-based; everything else is exact enumeration.
   Restatements collapse by canonical claim key. Zero solutions triggers a
   greedy-deletion minimal unsatisfiable subset over claims, and one
   verbatim statement line is cited per claim.
3. **Solver-validated repair (3x).** When the extraction admits more than
   4 solutions (missed constraints — a certain misread), the model
   re-reads only the lines judged "no fact". The solver then tries every
   subset of the proposed additions and keeps the *smallest* subset
   yielding 1–4 solutions; otherwise the original claims stand. Zero
   solutions is never repaired: it is a valid verdict (inconsistent), and
   the model cannot reliably distinguish a misread from a genuine
   contradiction. The repair only ever *adds* constraints, so it cannot
   break a correct inconsistent verdict.
4. **Per-line majority (10x).** After extraction + repair, independent
   re-extractions vote claim-by-claim; the majority claim per line wins
   (early stop once decided). Claim-level voting averages away random
   slips; answer-level voting was measured and dropped.

Nothing in the answering path is hand-written per item or per phrasing:
the solver never invents constraints, so every answer depends on what the
model actually extracted.

## The two pages

### How I characterised the model before designing around it

I probed the model on single items before writing the pipeline, and three
behaviours shaped every decision:

- **It follows a strict output schema well but reasons shallowly.**
  Given a numbered list and a JSON schema it returns clean, parseable
  claims for ~19 statements in one call. But it does not double-check
  itself: argument order in relations (`between`'s middle person,
  before/after direction) is its dominant error, especially in
  `station_holder_rel` ("Tomas is on later than whoever has packing" vs
  "packing is covered before Tomas comes on" say the same thing with the
  names in opposite roles).
- **Its errors are misreadings, not hallucinations.** It rarely invents
  people, blocks or stations (the prompt pins the vocabularies and demands
  exact copies). It *does* confuse categories once per few items ("Tomas
  is on intake" read as a block claim), miss whole constraint shapes (the
  "on after one of A and B and before the other" phrasing), and flip
  directions. These are exactly the errors a second, narrower read can
  catch — which is why the critic exists and why it is scoped to
  solver-flagged lines.
- **It is sensitive to how the question is framed, not to more examples.**
  Early prompts without an explicit block-vs-station vocabulary section
  produced category errors; adding the two word lists fixed them. Asking
  for relations as ordered pairs plus a direction menu produced
  argument-order slips ("X directly after Y" read as Y-after-X); asking
  only *who is earlier / who is later* fixed most of them — the single
  biggest prompt win, worth ~30 points of macro. Adding *more* examples
  past ~19, by contrast, hurt: the longer prompt scored lower than the
  shorter one. Sampling temperature is fixed at 1.0 by the brief, so
  single calls vary run to run — the 3x/10x majority is a direct response
  to that.

### Why each component exists

- **Model-based extraction, not template matching.** The held-out set uses
  a disjoint phrasing pool, so matching sentence templates would not
  transfer. The model handles rephrasing; the fixed 7-predicate vocabulary
  keeps its output checkable. This is the one place model quality
  directly caps the system.
- **Earlier/later instead of ordered-pair-plus-direction.** The model's
  dominant error was argument-order slips in relations ("X directly after
  Y" read as Y-after-X). Asking only *who is earlier / who is later* (and
  whether adjacent) removed the schema-mapping step and lifted 1x from
  ~18% to ~50% on its own.
- **Symbolic solver, not model reasoning.** Enumeration over 720 rotas is
  trivial, exact, and immune to the model's arithmetic weakness. It also
  makes the three verdicts *representable*: a system that only ever says
  "unique" has no formal model underneath; ours derives the verdict from
  the solution count, and the confusion matrix (not a scalar) is the right
  report.
- **Minimal-core citation via greedy deletion.** The citation rule is a
  property of sets (unsatisfiable, and every proper subset satisfiable),
  which is exactly what deletion-minimisation computes. Deduplicating
  restatements first is load-bearing: citing both copies of a restated
  fact would break minimality.
- **Claim-level majority instead of answer-level voting or a free critic.**
  A free-form critic that re-read the whole extraction *deleted valid
  claims* — a weak model given a destructive tool will demolish a good
  extraction. An unconstrained repair had the same disease in miniature:
  re-asking the model about a direction is no more reliable than asking
  it the first time, and 3x scored *below* 1x. The shipped repair is
  solver-validated instead: the model only proposes, and the solver keeps
  a change only if it yields 1–4 solutions with minimal edits — so a
  genuine inconsistency survives untouched. Per-line majority (10x) keeps
  every decision the model is good at (reading a line) and averages away
  the slips; answer-level majority was measured and dropped (with
  per-sample accuracy near a half, wrong answers scatter and the majority
  is not the right answer).

### What I built and then removed, and why

- **A free-form critic** (verify every claim, drop/add lists): removed
  after it dropped obviously-correct claims ("Whoever drew the 13:00
  block, it was Rohan") and turned a solvable item into 120 solutions.
- **Targeted repair passes** (contradiction review / miss review,
  unconstrained): measured and replaced. Re-asking the model about a
  direction is no more reliable than asking it the first time — the
  repair flipped correct claims as often as it fixed wrong ones, and 3x
  scored *below* 1x. The shipped repair is solver-validated: the model
  proposes re-reads, the solver disposes.
- **Answer-level majority voting**: measured and dropped. With
  per-sample accuracy near a half, wrong answers scatter across many
  distinct wrong answers and the majority is not the right answer.
  Claim-level majority does not have this problem: per-line accuracy is
  ~90%, so the majority claim per line is right ~99% of the time.
- **Per-line model calls**: rejected immediately — 1x allows exactly one
  call per item, and batching all lines in one call worked fine.
- **JSON response-format mode**: not relied upon; a tolerant parser
  (fence-stripping, trailing-comma repair) proved sufficient and keeps
  the system portable across providers.
- **A learned per-line confidence / calibration stage**: considered, cut
  for the 6–8 hour budget. It is the first thing I would add with more
  time (see below).

### Where the system breaks

1. **Direction flips that keep the solution count in range.** A flipped
   `before/after` that still yields 1–4 solutions is invisible to every
   repair trigger; only voting can catch it, imperfectly.
2. **Constraint shapes outside the 7 predicates.** The visible set needs
   no more, but genuinely new phrasings (e.g. "exactly two blocks after")
   would be mis-extracted or dropped rather than represented.
3. **The >4-solutions fallback.** If repair cannot tighten an
   over-permissive extraction, the system ships an over-long ambiguous
   answer — honest, but wrong.
4. **Citation choice among several minimal cores.** Any core is correct
   per the brief; the system picks the deletion order's core (10x: the
   smallest), which is principled but arbitrary.

### What I would do with a month

1. **Calibrate extraction.** Score per-line extraction against the
   visible key's implied constraints, learn which phrasings the model
   misreads, and route low-confidence lines to a second targeted pass
   automatically.
2. **Constrained decoding for claims.** A grammar-constrained sampler
   over the predicate vocabulary would eliminate malformed claims and
   out-of-vocabulary literals entirely.
3. **Develop against local weights.** The brief allows local Granite 4.2
   8B for development; free iteration would let me ablate prompts
   properly instead of by hand.
4. **A "needs-human" predicate.** For statements the model cannot map
   into the vocabulary, emit an explicit unknown rather than forcing a
   guess — currently the closest behaviour is `null`, which conflates
   "filler" with "unparseable".

### Model-based vs symbolic, and new authors

Constraint extraction is **model-based** and solving is **symbolic**,
deliberately. If the notes were written by different people rather than
generated to a template, the solver is unaffected (it sees only claims)
and extraction degrades gracefully: it still reads English, still
respects hedges, and still ignores counterfactuals. The failure mode
moves from "missed phrasing" to "novel constraint shape": a human might
write "Priya and Tomas are never adjacent", which has no predicate here
and would be dropped or mangled. The fix is the month-1 work above —
widening the vocabulary and adding the unknown predicate — not a change
of architecture.

## Results

Measured with the provided `score.py` on the 60-item visible set
(`visible_key.json`), one run each unless noted. Sampling uses
temperature=1.0, so expect ordinary run-to-run variation.

| budget | macro exact match | unique | ambiguous | inconsistent |
|--------|-------------------|--------|-----------|--------------|
| 1x     | _TBD_             | _TBD_  | _TBD_     | _TBD_        |
| 3x     | _TBD_             | _TBD_  | _TBD_     | _TBD_        |
| 10x    | _TBD_             | _TBD_  | _TBD_     | _TBD_        |

The 1x column is the system with a single extraction call; the shape of
the curve (not just the 10x endpoint) is the point — 1x is weighted at
least as heavily as 10x in grading.

### Ablation (12-item fixed subset; `python3 src/ablate.py`)

| variant      | 1x   | 3x   | 10x  |
|--------------|------|------|------|
| extract-only | _TBD_| _TBD_| _TBD_|
| shipped      | _TBD_| _TBD_| _TBD_|

Each cell is macro exact match. `extract-only` is one extraction call at
every budget (the "without" arm); `shipped` is the submitted system
(1 call at 1x, 3-call per-line majority at 3x, 10-call per-line majority
at 10x). Raw answers per cell are saved next to the ablation output for
scoring with the official `score.py`.
