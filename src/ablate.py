#!/usr/bin/env python3
"""Ablation: each system component with and without, at each budget.

Runs a fixed 12-item subset (4 of each kind) through two variants:
  extract-only : single extraction call, no voting (the "without" arm)
  shipped      : the submitted system -- 1 call at 1x, 3-call per-line
                 majority at 3x, 10-call per-line majority at 10x

Usage:
    python3 src/ablate.py --items <items.json> --key <visible_key.json> \\
        --out ablation_results/ablation.json [--workers N]

The subset is fixed by item id so re-runs are comparable. Sampling uses
temperature=1.0, so expect ordinary run-to-run variation.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extract import Budget, make_client  # noqa: E402
from src.pipeline import extract_once, majority_claims  # noqa: E402
from src.pipeline import answer_from_claims, validated_repair  # noqa: E402
from src.pipeline import _majority_decided  # noqa: E402
from src import csp  # noqa: E402

# fixed subset: 4 unique, 4 ambiguous, 4 inconsistent
SUBSET = ["B2-002", "B2-007", "B2-011", "B2-016",
          "B2-001", "B2-003", "B2-006", "B2-009",
          "B2-000", "B2-004", "B2-005", "B2-008"]


def run_variant(variant, budget_name, item, client):
    budget = Budget(budget_name)
    names, blocks, stations, holders = csp.parse_header(item["text"])
    lines = csp.body_lines(item["text"])
    iid = item["id"]
    if variant == "extract-only":
        claims = extract_once(client, budget, iid, names, blocks,
                              stations, holders, lines)
    elif budget_name == "1x":
        claims = extract_once(client, budget, iid, names, blocks,
                              stations, holders, lines)
    elif budget_name == "3x":
        claims = extract_once(client, budget, iid, names, blocks,
                              stations, holders, lines)
        claims, _ = validated_repair(client, budget, iid, names, blocks,
                                     stations, holders, lines, claims)
    else:  # 10x shipped
        claims = extract_once(client, budget, iid, names, blocks,
                              stations, holders, lines)
        claims, _ = validated_repair(client, budget, iid, names, blocks,
                                     stations, holders, lines, claims)
        samples = [claims]
        while budget.used < budget.cap:
            samples.append(extract_once(client, budget, iid, names, blocks,
                                        stations, holders, lines))
            if _majority_decided(samples):
                break
        claims = majority_claims(samples)
    ans = answer_from_claims(names, blocks, stations, holders, lines, claims)
    return iid, ans, budget.used


def macro(key, answers):
    per = {"unique": [0, 0], "ambiguous": [0, 0], "inconsistent": [0, 0]}
    for iid, k in key.items():
        if iid not in answers:
            continue
        case = k["case"]
        per[case][1] += 1
        if _exact(iid, k, answers[iid]):
            per[case][0] += 1
    rates = {c: (round(a / b, 4) if b else None) for c, (a, b) in per.items()}
    vals = [v for v in rates.values() if v is not None]
    return round(sum(vals) / len(vals), 4), rates


def _squash(s):
    return " ".join(str(s).split()).casefold()


def _core_match(cited, core):
    """Approximation of score.py's minimal-core check: same cardinality and
    a bijection between cited lines and a recorded core under normalised
    containment. Our citations are single verbatim lines, so the scorer's
    one-statement-per-citation rule is satisfied by construction."""
    if not isinstance(cited, list) or len(cited) != len(core):
        return False
    cs = [_squash(c) for c in cited]
    left = [_squash(w) for w in core]
    for c in cs:
        hit = next((w for w in left if w in c or c in w), None)
        if hit is None:
            return False
        left.remove(hit)
    return not left


def _exact(iid, k, ans):
    # minimal exact-match check mirroring score.py's rule
    case = k["case"]
    if ans.get("case") != case:
        return False
    if case == "unique":
        t = k["assignments"][0]
        g = ans.get("assignment", {})
        return (set(g) == set(t)
                and all(g[p].get(f) == v
                        for p, fs in t.items() for f, v in fs.items()))
    if case == "ambiguous":
        ts = k["assignments"]
        gs = ans.get("assignments", [])
        if len(ts) != len(gs):
            return False
        def norm(a):
            return tuple(sorted(
                (p, tuple(sorted(fs.items()))) for p, fs in a.items()))
        tn, gn = sorted(map(norm, ts)), sorted(map(norm, gs))
        return tn == gn
    if case == "inconsistent":
        cores = k.get("conflict_line_sets") or []
        return any(_core_match(ans.get("conflicts"), c) for c in cores)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    items = {i["id"]: i for i in json.load(open(args.items))}
    key = json.load(open(args.key))
    key = {iid: key[iid] for iid in SUBSET}
    items = [items[iid] for iid in SUBSET]
    client = make_client()

    table = {}
    for variant in ("extract-only", "shipped"):
        table[variant] = {}
        for budget_name in ("1x", "3x", "10x"):
            def one(item, _v=variant, _b=budget_name):
                return run_variant(_v, _b, item, client)
            answers = {}
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                for iid, ans, used in ex.map(one, items):
                    answers[iid] = ans
                    print(f"{variant} {budget_name} {iid}: "
                          f"{ans.get('case')} ({used} calls)", flush=True)
            m, rates = macro(key, answers)
            table[variant][budget_name] = {"macro_exact_match": m,
                                           "per_case_rate": rates}
            # save the raw answers so they can be scored with the official
            # score.py: <outdir>/answers_<variant>_<budget>.json
            adir = os.path.dirname(os.path.abspath(args.out))
            os.makedirs(adir, exist_ok=True)
            with open(os.path.join(
                    adir, f"answers_{variant}_{budget_name}.json"),
                    "w") as f:
                json.dump(answers, f, indent=1)
            print(f"== {variant} {budget_name}: {m} {rates}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(table, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")
    print(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
