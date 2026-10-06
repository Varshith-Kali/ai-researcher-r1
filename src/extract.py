"""Weak-model interface: extraction prompt, critic prompt, model client.

Every model call goes through `call_model`, which attaches the mandatory
X-Item-Id header and counts calls per item so the budget flags can be
enforced exactly (1x = exactly one call, 3x/10x = at most three/ten).
"""
import json
import os
import re

MODEL = "ibm-granite/granite-4.2-8b"
BASE_URL = "https://openrouter.ai/api/v1"

SYSTEM = (
    "You read shift notes from a facility and extract the definite scheduling "
    "facts. Reply with ONLY the requested JSON, no prose, no code fences."
)


def _read_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    # fall back to a .env file at the repo root (never committed)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in (".env", "env.txt"):
        p = os.path.join(here, name)
        if os.path.exists(p):
            for line in open(p):
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip()
    raise RuntimeError(
        "OPENROUTER_API_KEY not found: set it in the environment or in a "
        ".env file at the repo root (see .env.example).")


def make_client():
    """OpenAI-compatible client for the provided endpoint.

    In this sandbox the egress proxy URL breaks the vendored HTTP client at
    construction time, so we fall back to an explicitly configured client.
    In a normal environment the plain construction succeeds and is used.
    """
    from openai import OpenAI
    key = _read_key()
    try:
        return OpenAI(base_url=BASE_URL, api_key=key)
    except Exception:
        import httpx
        proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
        ca = "/run/hatch/egress-tls/ca-bundle.pem"
        http_client = httpx.Client(
            proxy=proxy, trust_env=False, timeout=300.0,
            verify=ca if os.path.exists(ca) else True)
        return OpenAI(base_url=BASE_URL, api_key=key, http_client=http_client)


class Budget:
    """Per-item call counter enforcing the budget flags."""

    def __init__(self, budget):
        self.cap = {"1x": 1, "3x": 3, "10x": 10}[budget]
        self.budget = budget
        self.used = 0

    def check(self):
        if self.used >= self.cap:
            raise RuntimeError(f"budget {self.budget} exceeded")


def call_model(client, budget, item_id, user_prompt, max_tokens=4000):
    budget.check()
    budget.used += 1
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": user_prompt}],
        temperature=1.0,
        top_p=0.95,
        max_tokens=max_tokens,
        extra_body={"reasoning": {"enabled": False}},
        extra_headers={"X-Item-Id": item_id},
    )
    return resp.choices[0].message.content


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #

EXTRACTION_PROMPT = """You read shift notes from a facility and extract the definite scheduling facts.

CONTEXT
- Staff on the rota: {names}
- BLOCK words (time blocks, in order): {blocks}. Each person works exactly one block, one person per block.
- STATION words: {stations}. The people who hold a station are: {holders}. Each of them holds exactly one station; everyone else holds no station.
- A sentence about someone being "on" a BLOCK word is a block fact. A sentence about someone being "on" a STATION word is a station fact. Never mix the two vocabularies: "on intake" is a station fact, never a block.

STATEMENTS (numbered; read each carefully):
{numbered}

TASK
For EACH statement, first decide its kind, then extract the fact:
- kind "block": the statement says who is on which time block.
- kind "station": the statement says who holds which station.
- kind "station_holder": the statement compares the block of whoever holds a station against someone's block ("whoever has <station> ...", "the person on <station> ...", "the <station> station is covered ...").
- kind "none": the statement asserts nothing about the current schedule.
- Hedged wording ("I'm fairly sure", "from what I recall", "speaking from memory", "going off the roster", "as far as I know", "my recollection is that") still states a definite fact. Treat it as certain.
- Restatements ("It bears repeating:", "Noted twice in the handover:", "This came up more than once, so:", "Worth restating, since it came up twice in handover:") state the same fact again. Extract the fact.
- "X lobbied / asked / put in for block B and was turned down / without success / did not get it" means X is NOT on block B.
- "X is the one who opens up at block B" / "X takes block B, as things stand" / "whoever drew block B, it was X" / "block B is X's" / "B is when X is scheduled" all mean X is on block B.
- For ORDER statements, decide only: WHO IS EARLIER and WHO IS LATER.
  "X is done before Y starts" / "X precedes Y" / "by the time Y starts, X has already been on" -> X earlier, Y later.
  "X comes later in the day than Y" / "X takes over from Y later in the day" -> Y earlier, X later.
  "X is on the block directly after Y" / "X relieves Y directly, with no block in between" / "Y hands straight over to X" / "Y then X, back to back" / "there is no block between Y's and X's, in that order" -> Y earlier, X later, ADJACENT (the very next block, nothing in between).
  "The <station> station is covered earlier in the day than P's block" / "whoever has <station> is done before P starts" -> the HOLDER is earlier than P.
  "P is on later than whoever has <station>" -> the HOLDER is earlier than P.
- For BETWEEN statements, decide only: WHO IS IN THE MIDDLE.
  "Put M between A and B" / "M is between A and B" / "M sits between A and B on the rota" / "M's block falls between A's and B's" / "M works at some point between A and B" / "M is on after one of A and B and before the other" -> mid is M; the other two names are "others" (their order does not matter).
- IGNORE a statement if it: talks about the past ("back in the old arrangement", "last month", "in the spring", "on the previous cycle", "was on <station>"); is a counterfactual or wish ("would have", "if X had", "putting X on", "had the rota gone the other way"); expresses uncertainty without asserting anything ("nobody could remember whether", "left open in the handover", "came up, but nothing was minuted", "some disagreement about whether"); or is about anything other than the current schedule (surveys, car-sharing, fire drills, equipment, visitors, catering, deliveries).

OUTPUT
Reply with ONLY a JSON object of this shape:
{{"claims": [{{"line": 0, "kind": "block|station|station_holder|none", "claim": {{...}} or null}}, ...]}}
One entry per statement, in order. "claim" is null when kind is "none".

A claim is exactly one of:
- {{"pred":"on_block","person":"<name>","block":"<block>"}}
- {{"pred":"not_on_block","person":"<name>","block":"<block>"}}
- {{"pred":"block_rel","earlier":"<name>","later":"<name>","adjacent":true/false}} (earlier's block is before later's; adjacent=true means the very next block, nothing in between)
- {{"pred":"between","mid":"<name>","others":["<name>","<name>"]}} (mid's block strictly between the other two's, either order)
- {{"pred":"on_station","person":"<name>","station":"<station>"}}
- {{"pred":"not_on_station","person":"<name>","station":"<station>"}}
- {{"pred":"station_holder_rel","station":"<station>","person":"<name>","holder":"earlier|later"}} (is the station HOLDER's block earlier or later than the person's block?)

Copy names, blocks and stations EXACTLY as written in CONTEXT. Do not invent values.

EXAMPLES
"Tomas is on intake." -> kind "station": {{"pred":"on_station","person":"Tomas","station":"intake"}}
"I'm fairly sure Priya is on calibration." -> kind "station": {{"pred":"on_station","person":"Priya","station":"calibration"}}
"If Samuel had taken the 11:00 block the handover would have been smoother." -> kind "none": null
"Whoever has packing is done before Daniel starts." -> kind "station_holder": {{"pred":"station_holder_rel","station":"packing","person":"Daniel","holder":"earlier"}}
"The calibration station is covered earlier in the day than Tomas's block." -> kind "station_holder": {{"pred":"station_holder_rel","station":"calibration","person":"Tomas","holder":"earlier"}}
"Tomas is on later than whoever has packing." -> kind "station_holder": {{"pred":"station_holder_rel","station":"packing","person":"Tomas","holder":"earlier"}}
"Daniel is on later than whoever has calibration." -> kind "station_holder": {{"pred":"station_holder_rel","station":"calibration","person":"Daniel","holder":"earlier"}}
"Put Meera between Tomas and Rohan, though not necessarily next to either." -> kind "block": {{"pred":"between","mid":"Meera","others":["Tomas","Rohan"]}}
"Whichever way round Meera and Nadia are, Rohan is between them." -> kind "block": {{"pred":"between","mid":"Rohan","others":["Meera","Nadia"]}}
"Rohan is on after one of Ayesha and Tomas and before the other." -> kind "block": {{"pred":"between","mid":"Rohan","others":["Ayesha","Tomas"]}}
"Nadia then Priya, back to back." -> kind "block": {{"pred":"block_rel","earlier":"Nadia","later":"Priya","adjacent":true}}
"Rohan is on the block directly after Meera." -> kind "block": {{"pred":"block_rel","earlier":"Meera","later":"Rohan","adjacent":true}}
"Meera hands straight over to Rohan." -> kind "block": {{"pred":"block_rel","earlier":"Meera","later":"Rohan","adjacent":true}}
"Rohan lobbied for 13:00 without success." -> kind "block": {{"pred":"not_on_block","person":"Rohan","block":"13:00"}}
"Tomas and Samuel car-share when their shifts allow." -> kind "none": null
"Lukas takes 15:00, as things stand." -> kind "block": {{"pred":"on_block","person":"Lukas","block":"15:00"}}
"Daniel is the one who opens up at 09:00." -> kind "block": {{"pred":"on_block","person":"Daniel","block":"09:00"}}
"Back in the old arrangement, Tomas took calibration." -> kind "none": null
"Meera would have been a better fit for the 13:00 block." -> kind "none": null
"""



def extraction_prompt(names, blocks, stations, holders, lines):
    numbered = "\n".join(f"{i}: {l}" for i, l in enumerate(lines))
    return EXTRACTION_PROMPT.format(
        names=", ".join(names), blocks=", ".join(blocks),
        stations=", ".join(stations), holders=", ".join(holders),
        numbered=numbered)


# --------------------------------------------------------------------------- #
# targeted review (3x repair)
#
# When the solver flags a degenerate extraction (0 or >4 solutions), we ask
# the model to re-read ONLY the suspect lines. The solver then searches over
# subsets of the proposed changes for the smallest one that yields 1-4
# solutions; if none does, the original claims stand. This keeps the repair
# conservative: a genuine inconsistency survives, a slip gets fixed.
# --------------------------------------------------------------------------- #

MISS_REVIEW_PROMPT = """The extracted facts admit {n_solutions} consistent schedules, but a sound extraction admits at most four: at least one statement below was wrongly judged to assert nothing.

CANDIDATE STATEMENTS (currently read as asserting nothing):
{candidates}

Re-read each. If it states a definite fact about the CURRENT shift, extract it (decide only WHO IS EARLIER / WHO IS LATER, or WHO IS IN THE MIDDLE). Leave it null only if it truly asserts nothing: talk of the past, counterfactuals, uncertainty that asserts nothing, or off-topic chatter.

Reply with ONLY a JSON object:
{{"claims": [{{"line": <n>, "claim": {{<pred>...}} or null}}, ...]}}
One entry per statement above, in order, using the same claim shapes as above.
"""

def miss_review_prompt(lines, n_solutions, candidate_line_idxs):
    candidates = "\n".join(f"{i}: {lines[i]}" for i in candidate_line_idxs)
    return MISS_REVIEW_PROMPT.format(n_solutions=n_solutions,
                                     candidates=candidates)


def review_claims_from_parsed(parsed, line_idxs):
    """{line_idx: claim|None} fresh claims for the reviewed lines."""
    out = {}
    if not isinstance(parsed, dict):
        return out
    items = parsed.get("claims")
    if not isinstance(items, list):
        return out
    for entry in items:
        if not isinstance(entry, dict):
            continue
        i = entry.get("line")
        if isinstance(i, int) and i in line_idxs:
            out[i] = normalize_claim(entry.get("claim"))
    return out


# --------------------------------------------------------------------------- #
# tolerant JSON parsing
# --------------------------------------------------------------------------- #

def parse_json_loose(text):
    """Best-effort parse of a JSON object from model output."""
    if not text:
        return None
    # strip code fences
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    blob = text[start:end + 1]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        pass
    # repair: drop trailing commas, drop // comments
    blob2 = re.sub(r",\s*([}\]])", r"\1", blob)
    blob2 = re.sub(r"(?m)^\s*//.*$", "", blob2)
    try:
        return json.loads(blob2)
    except json.JSONDecodeError:
        return None


def normalize_claim(c):
    """Map the model's earlier/later vocabulary to the solver's internal
    claim shapes. Accepts both the current and legacy shapes."""
    if not isinstance(c, dict) or "pred" not in c:
        return None
    c = dict(c)
    p = c["pred"]
    if p == "block_rel" and "earlier" in c:
        adj = c.get("adjacent")
        c = {"pred": "block_rel", "a": c["earlier"], "b": c["later"],
             "rel": "directly_before" if adj else "before"}
    elif p == "station_holder_rel" and "holder" in c:
        c = {"pred": "station_holder_rel", "station": c["station"],
             "person": c["person"],
             "rel": "before" if c["holder"] == "earlier" else "after"}
    elif p == "between" and "others" in c:
        others = c["others"]
        if not isinstance(others, list) or len(others) != 2:
            return None
        c = {"pred": "between", "mid": c["mid"],
             "a": others[0], "b": others[1]}
    return c


def claims_from_parsed(parsed, n_lines):
    """[claim|None] per line from a parsed extraction payload."""
    out = [None] * n_lines
    if not isinstance(parsed, dict):
        return out
    items = parsed.get("claims")
    if not isinstance(items, list):
        return out
    for entry in items:
        if not isinstance(entry, dict):
            continue
        i = entry.get("line")
        if isinstance(i, int) and 0 <= i < n_lines:
            c = normalize_claim(entry.get("claim"))
            out[i] = c
    return out
