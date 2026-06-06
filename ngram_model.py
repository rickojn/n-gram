"""
ngram_model.py  —  Word-level n-gram language model with recursive backoff.

Usage
-----
    python ngram_model.py <text_file> --n <order> [--temperature <float>]

Steps
-----
  1. Read a plain-text file.
  2. Build a vocabulary: word tokens + <BOS> + <EOS>  (newlines → <EOS>).
  3. Build n-gram count tables for every order from 1 up to n, using
     recursive (n-1)-gram backoff to fill gaps at generation time.
  4. Prompt the user for how many tokens to generate.
  5. Prompt the user for a seed string.
  6. Generate tokens by:
       a. Validating every prompt token against the vocab.
       b. Using the longest available context (up to n-1 tokens) to look
          up raw counts; backing off to shorter contexts until a non-zero
          row is found.
       c. Applying softmax (with temperature) over those counts to get a
          probability distribution, then sampling.
       d. Stopping on <EOS> or when the requested number of tokens is reached.
"""

import argparse
import math
import random
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ── special tokens ────────────────────────────────────────────────────────────
BOS = "<BOS>"
EOS = "<EOS>"


# ── tokenisation ─────────────────────────────────────────────────────────────

def tokenise(text: str) -> List[str]:
    """
    Split *text* into word-level tokens.
    Newlines become EOS tokens; other whitespace is ignored.
    Punctuation is split off as its own token.
    """
    tokens: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        # split on whitespace, then further split off leading/trailing punctuation
        words = re.findall(r"[A-Za-z0-9']+|[^A-Za-z0-9'\s]", line)
        tokens.extend(w.lower() for w in words if w.strip())
        tokens.append(EOS)          # newline → EOS
    # strip a trailing EOS that comes from the final newline of most files
    while tokens and tokens[-1] == EOS:
        tokens.pop()
    return tokens


def build_vocab(tokens: List[str]) -> Dict[str, int]:
    """Return {token: index} for every unique surface form plus BOS/EOS."""
    vocab = {BOS: 0, EOS: 1}
    for tok in tokens:
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


# ── n-gram tables ─────────────────────────────────────────────────────────────

# A table maps a context tuple → {next_token: count}
NGramTable = Dict[Tuple[str, ...], Dict[str, int]]


def build_ngram_tables(tokens: List[str], n: int) -> Dict[int, NGramTable]:
    """
    Build count tables for every order 1 … n.

    The corpus is broken into sentences delimited by EOS.  Each sentence is
    prepended with (n-1) BOS tokens so that the model can generate from the
    very start of a sequence.

    Returns a dict  {order: table}  where each table is
        { context_tuple : { next_token : count } }
    For a unigram table the context tuple is the empty tuple ().
    """
    # segment corpus into sentences
    sentences: List[List[str]] = []
    current: List[str] = []
    for tok in tokens:
        if tok == EOS:
            if current:
                sentences.append(current)
            current = []
        else:
            current.append(tok)
    if current:
        sentences.append(current)

    tables: Dict[int, NGramTable] = {order: defaultdict(lambda: defaultdict(int))
                                     for order in range(1, n + 1)}

    for sent in sentences:
        # pad with (n-1) BOS tokens at the start; close with EOS
        padded = [BOS] * (n - 1) + sent + [EOS]
        for i in range(n - 1, len(padded)):
            next_tok = padded[i]
            for order in range(1, n + 1):
                context = tuple(padded[i - order + 1: i])  # length = order-1
                tables[order][context][next_tok] += 1

    # convert inner defaultdicts to plain dicts for cleaner downstream use
    return {order: {ctx: dict(dist) for ctx, dist in table.items()}
            for order, table in tables.items()}


# ── probability / sampling ────────────────────────────────────────────────────

def softmax(values: List[float], temperature: float) -> List[float]:
    """
    Softmax with temperature.
    temperature > 1  → flatter (more random)
    temperature < 1  → sharper (more greedy)
    temperature = 1  → standard softmax
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    scaled = [v / temperature for v in values]
    max_v = max(scaled)
    exps = [math.exp(v - max_v) for v in scaled]   # numerical stability
    total = sum(exps)
    return [e / total for e in exps]


def sample(distribution: Dict[str, int], temperature: float) -> str:
    """Sample a token from a raw-count distribution via softmax."""
    tokens = list(distribution.keys())
    counts = [float(distribution[t]) for t in tokens]
    probs = softmax(counts, temperature)
    r = random.random()
    cumulative = 0.0
    for tok, p in zip(tokens, probs):
        cumulative += p
        if r <= cumulative:
            return tok
    return tokens[-1]   # guard against floating-point rounding


def get_distribution_with_backoff(
    context: List[str],
    tables: Dict[int, NGramTable],
    n: int,
) -> Optional[Dict[str, int]]:
    """
    Look up the distribution for *context* using the longest matching prefix,
    backing off to shorter contexts down to unigrams (empty context).

    Returns None only if even the unigram table is empty (should never happen).
    """
    # try from longest context (n-gram) down to unigram
    for order in range(n, 0, -1):
        ctx = tuple(context[-(order - 1):]) if order > 1 else ()
        dist = tables[order].get(ctx)
        if dist:
            return dist
    return None


# ── generation ────────────────────────────────────────────────────────────────

def generate(
    prompt_tokens: List[str],
    tables: Dict[int, NGramTable],
    n: int,
    max_tokens: int,
    temperature: float,
) -> List[str]:
    """
    Generate up to *max_tokens* tokens starting from *prompt_tokens*.
    Stops early on EOS.
    """
    context = list(prompt_tokens)   # running context window
    generated: List[str] = []

    while max_tokens <= 0 or len(generated) < max_tokens:
        dist = get_distribution_with_backoff(context, tables, n)
        if dist is None:
            print("[warning] No distribution found even after full backoff; stopping.")
            break

        next_tok = sample(dist, temperature)
        if next_tok == EOS:
            print("[EOS reached]")
            break

        generated.append(next_tok)
        context.append(next_tok)

    return generated


# ── CLI helpers ───────────────────────────────────────────────────────────────

def validate_prompt(prompt_tokens: List[str], vocab: Dict[str, int]) -> None:
    """Raise SystemExit if any token is outside the vocabulary."""
    unknown = [t for t in prompt_tokens if t not in vocab]
    if unknown:
        print(f"[error] The following prompt tokens are not in the vocabulary:\n"
              f"  {unknown}\n"
              f"Please revise your prompt and try again.", file=sys.stderr)
        sys.exit(1)


def prompt_int(msg: str) -> int:
    while True:
        raw = input(msg).strip()
        try:
            val = int(raw)
            if val >= 0:
                return val
            print("  Please enter a non-negative integer.")
        except ValueError:
            print("  Invalid input — please enter a whole number.")


def prompt_string(msg: str) -> str:
    """Return the user's input string, or an empty string to signal BOS."""
    return input(msg).strip()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Word-level n-gram language model with recursive backoff.")
    parser.add_argument("text_file",
                        help="Path to the plain-text corpus file.")
    parser.add_argument("--n", type=int, default=3,
                        help="N-gram order (default: 3 = trigram).")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Softmax temperature — higher = more random "
                             "(default: 1.0).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    if args.n < 1:
        print("[error] --n must be at least 1.", file=sys.stderr)
        sys.exit(1)

    if args.seed is not None:
        random.seed(args.seed)

    # ── 1. read corpus ────────────────────────────────────────────────────────
    try:
        with open(args.text_file, "r", encoding="utf-8") as fh:
            raw_text = fh.read()
    except FileNotFoundError:
        print(f"[error] File not found: {args.text_file}", file=sys.stderr)
        sys.exit(1)

    # ── 2. tokenise + build vocab ─────────────────────────────────────────────
    tokens = tokenise(raw_text)
    vocab  = build_vocab(tokens)

    print(f"\nCorpus stats")
    print(f"  Tokens  : {len(tokens):,}")
    print(f"  Vocab   : {len(vocab):,}  (incl. BOS / EOS)")
    print(f"  N-gram order: {args.n}")

    # ── 3. build n-gram tables ────────────────────────────────────────────────
    print("\nBuilding n-gram tables …", end=" ", flush=True)
    tables = build_ngram_tables(tokens, args.n)
    for order, table in tables.items():
        total_ngrams = sum(len(dist) for dist in table.values())
        print(f"\n  {order}-gram : {len(table):,} unique contexts, "
              f"{total_ngrams:,} unique n-grams", end="")
    print("\nDone.\n")

    # ── 4 & 5. user input ─────────────────────────────────────────────────────
    max_tokens = prompt_int("How many tokens to generate? ")

    while True:
        prompt_str = prompt_string("Enter prompt: ")
        if prompt_str == "q":
            break

        # empty input → treat as BOS (start of sequence)
        if not prompt_str:
            prompt_tokens = [BOS]
            print("  (no prompt given — seeding with <BOS>)")
        else:
            prompt_tokens = re.findall(r"[A-Za-z0-9']+|[^A-Za-z0-9'\s]", prompt_str)
            prompt_tokens = [t.lower() for t in prompt_tokens if t.strip()]

        # ── 6. validate + generate ────────────────────────────────────────────────
        validate_prompt(prompt_tokens, vocab)

        display_prompt = "<BOS>" if prompt_tokens == [BOS] else " ".join(prompt_tokens)
        print(f"\nGenerating up to {max_tokens} tokens "
            f"(temperature={args.temperature}) …"
            f"\nPrompt context: {display_prompt}\n")

        generated = generate(
            prompt_tokens=prompt_tokens,
            tables=tables,
            n=args.n,
            max_tokens=max_tokens,
            temperature=args.temperature,
        )

        # exclude BOS from the rendered output — it's a control token, not surface text
        visible_tokens = [t for t in prompt_tokens if t != BOS] + generated
        result = visible_tokens[0] if visible_tokens else ""
        for tok in visible_tokens[1:]:
            if re.match(r"^[^A-Za-z0-9']", tok):   # punctuation → no space
                result += tok
            else:
                result += " " + tok

        print("─" * 60)
        print(result)
        print("─" * 60)
        print(f"\n[{len(generated)} token(s) generated]")


if __name__ == "__main__":
    main()
