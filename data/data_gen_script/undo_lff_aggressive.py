"""
Generates aggressive UNDO Flip-Flop sequences for stress-testing.

Structure of each sequence:
  [Build Phase] → [Undo Phase] → [Tail Phase] → terminal r

The key invariant:
  - Sequence length is fixed (in-distribution, ≤ 50)
  - Undo depth is the controlled independent variable
  - The model was trained on sparse undo (P_U=0.05), so this is
    an OOD-in-complexity / ID-in-length stress test.

Min length for a given undo_depth D: 3D + 3
  - Build:  2*(D+1) tokens  [D+1 writes]
  - Undo:   D tokens        [D consecutive pops]
  - Terminal r: 1 token
Max undo_depth for length L: floor((L-3)/3)
  → For L=50: D_max = 15
"""

import argparse
import random
import torch

VOCAB = {'0': 0, '1': 1, 'r': 3, 'i': 4, 'w': 5, 'u': 6}
IGNORE_INDEX = -100


def max_undo_depth(length):
    return (length - 3) // 3


def generate_aggressive_undo_sequence(length, undo_depth):
    """
    Generate a single fixed-length sequence with a controlled deep undo chain.
    
    Args:
        length:     Fixed total token count.
        undo_depth: Number of consecutive undos in the undo phase.
                    Must satisfy: undo_depth <= max_undo_depth(length)
    
    Returns:
        seq_ids, label_ids as torch.LongTensor
    """
    if undo_depth > max_undo_depth(length):
        raise ValueError(
            f"undo_depth={undo_depth} exceeds max={max_undo_depth(length)} "
            f"for length={length}. Minimum length needed: {3*undo_depth+3}"
        )
    if undo_depth < 1:
        raise ValueError("undo_depth must be >= 1")

    seq_tokens = []
    labels = []
    state_stack = []

    # ── Phase 1: Build ─────────────────────────────────────────────────────────
    # Push (undo_depth + 1) values so the undo phase never empties the stack.
    for _ in range(undo_depth + 1):
        v = str(random.choice(["0", "1"]))
        seq_tokens.extend(['w', v])
        labels.extend([IGNORE_INDEX, IGNORE_INDEX])
        state_stack.append(v)

    current_len = 2 * (undo_depth + 1)

    # ── Phase 2: Undo ──────────────────────────────────────────────────────────
    # Consecutive pops. Stack is guaranteed to have >= 1 element after each pop.
    for _ in range(undo_depth):
        seq_tokens.append('u')
        labels.append(IGNORE_INDEX)
        state_stack.pop()
        current_len += 1

    # Stack now has exactly 1 element: the ground truth for the terminal r.

    # ── Phase 3: Tail ──────────────────────────────────────────────────────────
    # Fill remaining budget with 'i' tokens (pure noise, no state change).
    # Budget: length - current_len - 1 (reserve 1 for terminal r)
    tail_budget = length - current_len - 1
    assert tail_budget >= 0, "Budget underflow — should be caught by undo_depth validation"

    # If tail_budget is odd, absorb one token with an 'r' checkpoint.
    # This gives an extra label (same ground truth as terminal r, since
    # 'i' tokens don't affect state) — useful signal, not harmful noise.
    if tail_budget % 2 == 1:
        seq_tokens.append('r')
        labels.append(int(state_stack[-1]))
        tail_budget -= 1

    # Fill the rest with 'i v' pairs
    while tail_budget > 0:
        val = str(random.choice(["0", "1"]))
        seq_tokens.extend(['i', val])
        labels.extend([IGNORE_INDEX, IGNORE_INDEX])
        tail_budget -= 2

    # ── Terminal r ─────────────────────────────────────────────────────────────
    seq_tokens.append('r')
    labels.append(int(state_stack[-1]))

    assert len(seq_tokens) == length, (
        f"Length mismatch: generated {len(seq_tokens)}, expected {length}"
    )

    seq_ids = torch.tensor(
        [VOCAB[t] if t in VOCAB else int(t) for t in seq_tokens],
        dtype=torch.long
    )
    label_ids = torch.tensor(labels, dtype=torch.long)

    return seq_ids, label_ids


def main(args):
    if args.undo_depth_max is None:
        args.undo_depth_max = args.undo_depth_min

    d_max_for_length = max_undo_depth(args.length)
    if args.undo_depth_max > d_max_for_length:
        raise ValueError(
            f"undo_depth_max={args.undo_depth_max} exceeds max allowable depth "
            f"({d_max_for_length}) for length={args.length}. "
            f"Either reduce undo_depth_max or increase --length."
        )

    dataset = []
    for _ in range(args.num_samples):
        d = random.randint(args.undo_depth_min, args.undo_depth_max)
        seq, label = generate_aggressive_undo_sequence(args.length, d)
        dataset.append({
            'input': seq,
            'label': label,
            'undo_depth': d        # stored for per-depth analysis
        })

    torch.save(dataset, args.output_file)
    print(
        f"Saved {args.num_samples} sequences to {args.output_file}\n"
        f"  Length:      {args.length} (fixed, ID range)\n"
        f"  Undo depth:  {args.undo_depth_min}–{args.undo_depth_max} "
        f"(max for this length: {d_max_for_length})"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate aggressive UNDO Flip-Flop stress-test dataset"
    )
    parser.add_argument('--num_samples',     type=int, required=True)
    parser.add_argument('--length',          type=int, required=True,
                        help="Fixed sequence length. Keep ≤ 50 for ID range.")
    parser.add_argument('--undo_depth_min',  type=int, required=True,
                        help="Minimum consecutive undo depth.")
    parser.add_argument('--undo_depth_max',  type=int, default=None,
                        help="Maximum consecutive undo depth (default: same as min).")
    parser.add_argument('--output_file',     type=str, required=True)
    args = parser.parse_args()
    main(args)


# ── Recommended call pattern for a sweep ──────────────────────────────────────
#
# Single fixed depth (e.g., D=10, L=50):
#   python undo_lff_aggressive.py --num_samples 2000 --length 50
#       --undo_depth_min 10 --output_file aggressive_d10.pt
#
# Sweep D=1..15 in a single file (for per-depth breakdown via 'undo_depth' key):
#   python undo_lff_aggressive.py --num_samples 2000 --length 50
#       --undo_depth_min 1 --undo_depth_max 15 --output_file aggressive_sweep.pt
#
# For a clean depth-vs-accuracy curve, generate one file per depth:
#   for D in 1 3 5 7 10 12 15; do
#     python undo_lff_aggressive.py --num_samples 500 --length 50
#         --undo_depth_min $D --output_file aggressive_d${D}.pt
#   done