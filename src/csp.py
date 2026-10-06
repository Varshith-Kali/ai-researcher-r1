"""Symbolic constraint solver for shift-note items.

The weak model extracts one formal claim per statement line (see extract.py).
Everything from here on is deterministic: enumerate every possible rota,
evaluate the claims, and classify the item as unique / ambiguous / inconsistent.

For inconsistent items we compute a minimal unsatisfiable subset of claims
(greedy deletion) and cite one verbatim statement line per claim.
"""
import itertools
import re


# --------------------------------------------------------------------------- #
# header parsing
# --------------------------------------------------------------------------- #

def parse_header(text):
    """Parse the first line of an item into (names, blocks, stations, holders)."""
    header = text.split("\n")[0]
    m = re.search(r"rota:\s*(.+?)\.\s*Blocks", header)
    names = [x.strip() for x in m.group(1).split(",")]
    m = re.search(r"Blocks run\s*(.+?),\s*one person per block", header)
    blocks = [x.strip() for x in m.group(1).split(",")]
    m = re.search(r"one person on each:\s*(.+?)\.", header)
    stations = [x.strip() for x in m.group(1).split(",")]
    m = re.search(r"people on a station are\s*(.+?);", header)
    holders = [x.strip() for x in m.group(1).split(",")]
    return names, blocks, stations, holders


def body_lines(text):
    """The numbered statement lines (everything after the header)."""
    return [l for l in (s.strip() for s in text.split("\n")[1:]) if l]


# --------------------------------------------------------------------------- #
# claim model
# --------------------------------------------------------------------------- #

def canon_key(claim):
    """Canonical dedup key. Restatements that reorder arguments collapse."""
    p = claim["pred"]
    if p == "on_block":
        return (p, claim["person"], claim["block"])
    if p == "not_on_block":
        return (p, claim["person"], claim["block"])
    if p == "block_rel":
        a, b, rel = claim["a"], claim["b"], claim["rel"]
        # normalise to a single direction so (a after b) == (b before a)
        if rel == "after":
            a, b, rel = b, a, "before"
        elif rel == "directly_after":
            a, b, rel = b, a, "directly_before"
        return (p, a, b, rel)
    if p == "between":
        return (p, claim["mid"], tuple(sorted((claim["a"], claim["b"]))))
    if p == "on_station":
        return (p, claim["person"], claim["station"])
    if p == "not_on_station":
        return (p, claim["person"], claim["station"])
    if p == "station_holder_rel":
        return (p, claim["station"], claim["person"], claim["rel"])
    raise ValueError(f"unknown predicate {p!r}")


def valid_claim(claim, names, blocks, stations):
    """A claim is usable when every literal it names comes from the header."""
    try:
        p = claim["pred"]
        if p in ("on_block", "not_on_block"):
            return claim["person"] in names and claim["block"] in blocks
        if p == "block_rel":
            return (claim["a"] in names and claim["b"] in names
                    and claim["rel"] in ("before", "after",
                                         "directly_before", "directly_after"))
        if p == "between":
            return (claim["mid"] in names and claim["a"] in names
                    and claim["b"] in names
                    and len({claim["mid"], claim["a"], claim["b"]}) == 3)
        if p in ("on_station", "not_on_station"):
            return claim["person"] in names and claim["station"] in stations
        if p == "station_holder_rel":
            return (claim["station"] in stations and claim["person"] in names
                    and claim["rel"] in ("before", "after"))
        return False
    except (KeyError, TypeError):
        return False


def check(claim, names, blocks, stations, holders, block_of, station_of):
    """Evaluate one claim under a candidate (block_of, station_of) assignment."""
    bi = {b: i for i, b in enumerate(blocks)}
    p = claim["pred"]
    if p == "on_block":
        return block_of[claim["person"]] == bi[claim["block"]]
    if p == "not_on_block":
        return block_of[claim["person"]] != bi[claim["block"]]
    if p == "block_rel":
        a, b = block_of[claim["a"]], block_of[claim["b"]]
        rel = claim["rel"]
        if rel == "before":
            return a < b
        if rel == "after":
            return a > b
        if rel == "directly_before":
            return a == b - 1
        if rel == "directly_after":
            return a == b + 1
        raise ValueError(rel)
    if p == "between":
        m_, a, b = (block_of[claim["mid"]],
                    block_of[claim["a"]], block_of[claim["b"]])
        return (a < m_ < b) or (b < m_ < a)
    if p == "on_station":
        # a non-holder can never hold a station: the claim is unsatisfiable
        if claim["person"] not in holders:
            return False
        return station_of[claim["person"]] == claim["station"]
    if p == "not_on_station":
        if claim["person"] not in holders:
            return True
        return station_of[claim["person"]] != claim["station"]
    if p == "station_holder_rel":
        holder = next(h for h, s in station_of.items()
                      if s == claim["station"])
        hb, pb = block_of[holder], block_of[claim["person"]]
        return (hb < pb) if claim["rel"] == "before" else (hb > pb)
    raise ValueError(f"unknown predicate {p!r}")


# --------------------------------------------------------------------------- #
# solving
# --------------------------------------------------------------------------- #

def solve_all(names, blocks, stations, holders, claims):
    """Every (person -> block, holder -> station) pair satisfying all claims."""
    sols = []
    n = len(names)
    for bperm in itertools.permutations(range(n)):
        block_of = dict(zip(names, bperm))
        for sperm in itertools.permutations(stations):
            station_of = dict(zip(holders, sperm))
            if all(check(c, names, blocks, stations, holders,
                         block_of, station_of) for c in claims):
                sols.append((block_of, station_of))
    return sols


def dedupe(claims_with_lines):
    """Collapse restatements: one entry per distinct claim, keeping line idxs."""
    seen = {}
    for line_idx, claim in claims_with_lines:
        k = canon_key(claim)
        if k in seen:
            seen[k]["lines"].append(line_idx)
        else:
            seen[k] = {"claim": claim, "lines": [line_idx]}
    return list(seen.values())


def is_sat(names, blocks, stations, holders, entries, skip=None):
    """Do the entries (minus optional skip idx) admit any solution?"""
    claims = [e["claim"] for i, e in enumerate(entries) if i != skip]
    return bool(solve_all(names, blocks, stations, holders, claims))


def minimal_core(names, blocks, stations, holders, entries):
    """Greedy-deletion minimal unsatisfiable subset of deduped claim entries."""
    assert not is_sat(names, blocks, stations, holders, entries)
    core = list(range(len(entries)))
    changed = True
    while changed:
        changed = False
        for i in list(core):
            trial = [e for j, e in enumerate(entries) if j in core and j != i]
            if not solve_all(names, blocks, stations, holders,
                             [e["claim"] for e in trial]):
                core.remove(i)
                changed = True
    return [entries[i] for i in core]


def assignment_json(names, blocks, holders, block_of, station_of):
    out = {}
    for person in names:
        entry = {"block": blocks[block_of[person]]}
        if person in holders:
            entry["station"] = station_of[person]
        out[person] = entry
    return out


def answer_for(names, blocks, stations, holders, lines, entries):
    """Classify + build the submission entry from deduped claim entries."""
    claims = [e["claim"] for e in entries]
    sols = solve_all(names, blocks, stations, holders, claims)
    if len(sols) == 1:
        block_of, station_of = sols[0]
        return {"case": "unique",
                "assignment": assignment_json(names, blocks, holders,
                                              block_of, station_of)}
    if 2 <= len(sols) <= 4:
        return {"case": "ambiguous",
                "assignments": [assignment_json(names, blocks, holders, b, s)
                                for b, s in sols]}
    if len(sols) == 0:
        core = minimal_core(names, blocks, stations, holders, entries)
        # one verbatim line per claim; first line wins for restatements
        return {"case": "inconsistent",
                "conflicts": [lines[e["lines"][0]] for e in core]}
    # Over-permissive extraction (missed constraints) is an extraction
    # failure, not a fourth kind. The repair/voting stages try to avoid
    # shipping this; if it still happens, cap at 4 to keep the output
    # format-valid (the item scores 0 either way -- the extraction was
    # wrong). We take the first 4 in enumeration order.
    sols = sols[:4]
    return {"case": "ambiguous",
            "assignments": [assignment_json(names, blocks, holders, b, s)
                            for b, s in sols]}
