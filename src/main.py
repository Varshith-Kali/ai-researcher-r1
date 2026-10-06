#!/usr/bin/env python3
"""Entrypoint for the shift-notes harness.

Usage:
    ./run <items.json> --budget <1x|3x|10x> --out <answers.json>

Reads the model API key from OPENROUTER_API_KEY or a .env file at the repo
root (see .env.example; the key is never committed).
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extract import Budget, make_client  # noqa: E402
from src.pipeline import solve_item_text  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items", help="path to items.json")
    ap.add_argument("--budget", required=True, choices=["1x", "3x", "10x"])
    ap.add_argument("--out", required=True, help="path to write answers.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="only run the first N items (debugging)")
    ap.add_argument("--save-claims", default="",
                    help="also dump per-item extracted claims to this JSON")
    args = ap.parse_args()

    with open(args.items) as f:
        items = json.load(f)
    if args.limit:
        items = items[:args.limit]

    client = make_client()

    def one(item):
        budget = Budget(args.budget)
        try:
            ans, claims = solve_item_text(item["id"], item["text"],
                                          client, budget)
        except Exception as e:  # never leave an item unanswered
            ans, claims = ({"case": "unique", "assignment": {}}, [])
        ans.pop("_error", None)
        return item["id"], ans, claims, budget.used

    answers = {}
    all_claims = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for item_id, ans, claims, used in ex.map(one, items):
            answers[item_id] = ans
            all_claims[item_id] = claims
            print(f"{item_id}: {ans.get('case')} ({used} calls)",
                  flush=True)

    with open(args.out, "w") as f:
        json.dump(answers, f, indent=1)
    print(f"wrote {args.out}")
    if args.save_claims:
        with open(args.save_claims, "w") as f:
            json.dump(all_claims, f, indent=1)
        print(f"wrote {args.save_claims}")


if __name__ == "__main__":
    main()
