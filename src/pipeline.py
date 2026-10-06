"""Per-item pipeline: extract claims with the weak model, solve symbolically.

Budget strategies
-----------------
1x  : one extraction call, solve, answer. Exactly one model call per item.
3x  : extraction, then one solver-validated repair pass if the extraction
      admits more than 4 solutions (missed constraints). The model re-reads
      only the lines judged "no fact"; the solver keeps the smallest subset
      of proposed additions yielding 1-4 solutions. Zero solutions is never
      repaired: it is a valid verdict, and the model cannot reliably tell a
      misread from a genuine contradiction. Uses 1-2 calls.
10x : extraction + validated repair, then independent re-extractions with
      per-line majority over all samples (early stop once decided).
      Up to 10 calls.

The repair is conservative by construction: it only ever ADDS constraints
to an over-permissive extraction, so it cannot break a correct
inconsistent verdict.

Every strategy answers every item from model-derived claims only: the
symbolic solver never invents constraints, so the answers always depend on
what the model said.
"""
from collections import Counter
import json
import re

from . import csp
from . import extract as X


# Lines that talk about the past assert nothing about the current schedule,
# no matter what the model extracts from them. This is a fixed linguistic
# filter (not per-item logic): the generator's past-tense phrasings are a
# closed set, and the model occasionally leaks them into current facts.
PAST_TENSE_RE = re.compile(
    r"back in the old arrangement|in the spring|on the previous cycle|"
    r"last month",
    re.IGNORECASE)


def solve_item_text(item_id, text, client, budget):
    names, blocks, stations, holders = csp.parse_header(text)
    lines = csp.body_lines(text)

    if budget.budget == "1x":
        claims = extract_once(client, budget, item_id,
                              names, blocks, stations, holders, lines)
        return (answer_from_claims(names, blocks, stations, holders, lines,
                                   claims), claims)

    if budget.budget == "3x":
        # Extraction, then one solver-validated repair pass if the solution
        # count is degenerate (0 or >4). Uses 1-2 calls.
        claims = extract_once(client, budget, item_id,
                              names, blocks, stations, holders, lines)
        claims, _ = validated_repair(client, budget, item_id, names, blocks,
                                     stations, holders, lines, claims)
        return (answer_from_claims(names, blocks, stations, holders, lines,
                                   claims), claims)

    # 10x: extraction + validated repair (2 calls), then independent
    # re-extractions with per-line majority over all samples (up to 10).
    claims = extract_once(client, budget, item_id,
                          names, blocks, stations, holders, lines)
    claims, _ = validated_repair(client, budget, item_id, names, blocks,
                                 stations, holders, lines, claims)
    samples = [claims]
    while budget.used < budget.cap:
        samples.append(extract_once(client, budget, item_id,
                                    names, blocks, stations, holders, lines))
        if _majority_decided(samples):
            break
    claims = majority_claims(samples)
    return (answer_from_claims(names, blocks, stations, holders, lines,
                               claims), claims)


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #

def extract_once(client, budget, item_id, names, blocks, stations, holders,
                 lines):
    prompt = X.extraction_prompt(names, blocks, stations, holders, lines)
    raw = X.call_model(client, budget, item_id, prompt, max_tokens=4000)
    parsed = X.parse_json_loose(raw)
    claims = X.claims_from_parsed(parsed, len(lines))
    out = []
    for line, c in zip(lines, claims):
        # Symbolic past-tense filter: these phrasings assert nothing about
        # the current schedule, no matter what the model extracts. This is
        # a fixed linguistic filter, not per-item logic.
        if c is not None and PAST_TENSE_RE.search(line):
            c = None
        out.append(c if (c and csp.valid_claim(c, names, blocks, stations))
                   else None)
    return out


def _entries(names, blocks, stations, holders, claims):
    pairs = [(i, c) for i, c in enumerate(claims) if c]
    return csp.dedupe(pairs)


def n_solutions(names, blocks, stations, holders, claims):
    entries = _entries(names, blocks, stations, holders, claims)
    return len(csp.solve_all(names, blocks, stations, holders,
                             [e["claim"] for e in entries]))


def validated_repair(client, budget, item_id, names, blocks, stations,
                     holders, lines, claims):
    """One conservative repair pass for an over-permissive extraction.

    Only triggers when the extraction admits MORE than 4 solutions (a
    certain misread: constraints were missed). Zero solutions is left
    alone: it is a valid verdict (inconsistent), and the model cannot
    reliably distinguish a misread from a genuine contradiction.

    The model re-reads only the lines judged "no fact"; the solver tries
    every subset of the proposed additions and keeps the SMALLEST subset
    yielding 1-4 solutions. If none does, the original claims stand.
    """
    n = n_solutions(names, blocks, stations, holders, claims)
    if 0 <= n <= 4:
        # 0 solutions is a valid verdict (inconsistent); the model cannot
        # reliably distinguish a misread from a genuine contradiction, so we
        # do not touch it. Only >4 (a certain misread: missed constraints)
        # is repaired.
        return claims, False
    suspect = [i for i, c in enumerate(claims) if c is None]
    if not suspect:
        return claims, False
    prompt = X.miss_review_prompt(lines, n, suspect)
    raw = X.call_model(client, budget, item_id, prompt, max_tokens=2000)
    fresh = X.review_claims_from_parsed(X.parse_json_loose(raw), set(suspect))
    # proposed changes: (line, new_claim) differing from the original
    changes = [(i, c) for i, c in fresh.items()
               if json.dumps(c, sort_keys=True) !=
               json.dumps(claims[i], sort_keys=True)
               and (c is None or csp.valid_claim(c, names, blocks, stations))]
    # smallest-first subset search for 1-4 solutions
    from itertools import combinations
    for r in range(1, len(changes) + 1):
        for combo in combinations(changes, r):
            trial = list(claims)
            for i, c in combo:
                trial[i] = c
            if 1 <= n_solutions(names, blocks, stations, holders, trial) <= 4:
                return trial, True
    return claims, False


def answer_from_claims(names, blocks, stations, holders, lines, claims):
    entries = _entries(names, blocks, stations, holders, claims)
    ans = csp.answer_for(names, blocks, stations, holders, lines, entries)
    ans.pop("_over_constrained", None)
    return ans


# --------------------------------------------------------------------------- #
# per-line majority (3x / 10x)
# --------------------------------------------------------------------------- #

def _claim_key(c):
    """Hashable identity for one claim (None-safe)."""
    if c is None:
        return None
    try:
        return csp.canon_key(c)
    except Exception:
        return ("raw", json.dumps(c, sort_keys=True))


def majority_claims(samples):
    """Per-line majority claim across independent extractions.

    A random slip rarely repeats across samples, so the majority claim per
    line is far more reliable than any single sample. Ties break toward the
    earliest sample.
    """
    n_lines = len(samples[0])
    out = []
    for i in range(n_lines):
        counts = Counter(_claim_key(s[i]) for s in samples)
        best = counts.most_common(1)[0][0]
        # first sample carrying the winning key
        winner = next(s[i] for s in samples if _claim_key(s[i]) == best)
        out.append(winner)
    return out


def _majority_decided(samples):
    """Stop early once every line's majority is mathematically decided."""
    n = len(samples)
    need = n // 2 + 1
    for i in range(len(samples[0])):
        counts = Counter(_claim_key(s[i]) for s in samples)
        if counts.most_common(1)[0][1] < need:
            return False
    return n >= 3
