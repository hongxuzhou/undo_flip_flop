"""
analyse_undo_dataset.py — Statistical analysis of aggressive UNDO FF datasets.

Covers:
  1. Global token distribution
  2. Undo depth distribution
  3. Phase composition (build / undo / tail per sequence)
  4. Label statistics (ground truth balance, r-token density)
  5. Stack depth trajectory statistics
  6. Optional matplotlib plots (--plot)

Usage:
  python analyse_undo_dataset.py --input aggressive_sweep.pt
  python analyse_undo_dataset.py --input aggressive_sweep.pt --plot
  python analyse_undo_dataset.py --input a.pt b.pt --labels "D=5" "D=10" --plot
"""

import argparse
import collections
import math
import sys
from pathlib import Path

import torch

# ── Optional matplotlib ────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Token vocab (inverse) ──────────────────────────────────────────────────────
ID2TOK = {0: '0', 1: '1', 3: 'r', 4: 'i', 5: 'w', 6: 'u'}
IGNORE_INDEX = -100


# ══════════════════════════════════════════════════════════════════════════════
# Core analysis
# ══════════════════════════════════════════════════════════════════════════════

def decode(seq_ids: torch.Tensor) -> list[str]:
    return [ID2TOK.get(int(t), str(int(t))) for t in seq_ids]


def analyse_sequence(item: dict) -> dict:
    """
    Extract per-sequence statistics. Returns a flat dict of scalars.
    """
    tokens = decode(item['input'])
    labels = item['label'].tolist()
    n = len(tokens)

    # ── Token counts ──────────────────────────────────────────────────────────
    counter = collections.Counter(tokens)

    # ── Phase boundaries (inferred by re-simulating the stack) ───────────────
    # Build phase: initial consecutive 'w v' block
    # Undo phase: first consecutive 'u' block after build
    # Tail phase: everything after undo until terminal 'r'
    stack = []
    build_end = 0       # exclusive index where build ends
    undo_start = None
    undo_end = None
    max_stack_depth = 0

    i = 0
    # Build phase: collect leading w-tokens
    while i < n - 1:
        if tokens[i] == 'w':
            stack.append(tokens[i + 1])
            max_stack_depth = max(max_stack_depth, len(stack))
            i += 2
        else:
            build_end = i
            break
    else:
        build_end = i  # edge case: sequence is all writes

    # Undo phase: consecutive 'u' tokens immediately after build
    if i < n and tokens[i] == 'u':
        undo_start = i
        while i < n and tokens[i] == 'u':
            i += 1
        undo_end = i

    actual_undo_depth = (undo_end - undo_start) if undo_start is not None else 0
    tail_start = undo_end if undo_end is not None else build_end
    tail_len = n - tail_start

    # ── Label / r-token stats ─────────────────────────────────────────────────
    r_positions = [j for j, t in enumerate(tokens) if t == 'r']
    label_values = [labels[j] for j in r_positions if labels[j] != IGNORE_INDEX]
    n_labels = len(label_values)
    n_label_1 = sum(label_values)
    n_label_0 = n_labels - n_label_1

    # ── Stored undo depth (from metadata if present) ──────────────────────────
    stored_depth = item.get('undo_depth', None)

    # ── Stack depth trajectory stats ─────────────────────────────────────────
    # Re-simulate full sequence for min/max stack depth
    stack2 = []
    depths = []
    j = 0
    while j < n:
        t = tokens[j]
        if t == 'w' and j + 1 < n:
            stack2.append(tokens[j + 1])
            j += 2
        elif t == 'u' and stack2:
            stack2.pop()
            j += 1
        elif t == 'i' and j + 1 < n:
            j += 2
        else:
            j += 1
        depths.append(len(stack2))

    return {
        'length':            n,
        'stored_depth':      stored_depth,
        'actual_undo_depth': actual_undo_depth,
        'build_tokens':      build_end,
        'undo_tokens':       actual_undo_depth,
        'tail_tokens':       tail_len,
        'max_stack_depth':   max(depths) if depths else 0,
        'min_stack_depth':   min(depths) if depths else 0,
        'n_w':               counter['w'],
        'n_u':               counter['u'],
        'n_i':               counter['i'],
        'n_r':               counter['r'],
        'n_labels':          n_labels,
        'n_label_0':         n_label_0,
        'n_label_1':         n_label_1,
        'frac_u':            counter['u'] / n,
        'frac_i':            counter['i'] / n,
    }


def analyse_dataset(path: str) -> tuple[list[dict], dict]:
    """Load a .pt file and return (per_seq_stats, aggregate_stats)."""
    data = torch.load(path, weights_only=False)
    if not data:
        raise ValueError(f"Empty dataset: {path}")

    per_seq = [analyse_sequence(item) for item in data]
    N = len(per_seq)

    def mean(key):
        return sum(s[key] for s in per_seq if s[key] is not None) / N

    def std(key):
        m = mean(key)
        return math.sqrt(sum((s[key] - m) ** 2 for s in per_seq if s[key] is not None) / N)

    def counts(key):
        return collections.Counter(
            s[key] for s in per_seq if s[key] is not None
        )

    # undo depth from metadata (may be None for original-format datasets)
    has_meta = all(s['stored_depth'] is not None for s in per_seq)

    agg = {
        'N':                    N,
        'has_meta':             has_meta,
        'length_mean':          mean('length'),
        'length_std':           std('length'),
        'length_dist':          counts('length'),
        'undo_depth_mean':      mean('actual_undo_depth'),
        'undo_depth_std':       std('actual_undo_depth'),
        'undo_depth_dist':      counts('actual_undo_depth'),
        'stored_depth_dist':    counts('stored_depth') if has_meta else None,
        'build_mean':           mean('build_tokens'),
        'undo_mean':            mean('undo_tokens'),
        'tail_mean':            mean('tail_tokens'),
        'max_stack_mean':       mean('max_stack_depth'),
        'max_stack_std':        std('max_stack_depth'),
        'frac_u_mean':          mean('frac_u'),
        'frac_i_mean':          mean('frac_i'),
        'total_labels':         sum(s['n_labels'] for s in per_seq),
        'total_label_0':        sum(s['n_label_0'] for s in per_seq),
        'total_label_1':        sum(s['n_label_1'] for s in per_seq),
        'per_seq':              per_seq,
    }
    return per_seq, agg


# ══════════════════════════════════════════════════════════════════════════════
# Pretty printing
# ══════════════════════════════════════════════════════════════════════════════

def _bar(value, max_value, width=30) -> str:
    filled = round(width * value / max_value) if max_value > 0 else 0
    return '█' * filled + '░' * (width - filled)


def print_report(path: str, agg: dict):
    sep = '─' * 60
    print(f"\n{'═'*60}")
    print(f"  Dataset: {Path(path).name}")
    print(f"  N = {agg['N']:,} sequences")
    print(f"{'═'*60}")

    # ── Length ────────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  SEQUENCE LENGTH")
    print(sep)
    print(f"  Mean ± Std : {agg['length_mean']:.1f} ± {agg['length_std']:.2f}")
    unique_lengths = sorted(agg['length_dist'])
    if len(unique_lengths) == 1:
        print(f"  Fixed length: {unique_lengths[0]}")
    else:
        print(f"  Range: {min(unique_lengths)} – {max(unique_lengths)}")

    # ── Undo depth ────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  UNDO DEPTH DISTRIBUTION  (actual consecutive u-tokens)")
    print(sep)
    print(f"  Mean ± Std : {agg['undo_depth_mean']:.2f} ± {agg['undo_depth_std']:.2f}")
    depth_dist = agg['undo_depth_dist']
    max_count = max(depth_dist.values()) if depth_dist else 1
    for d in sorted(depth_dist):
        c = depth_dist[d]
        pct = 100 * c / agg['N']
        bar = _bar(c, max_count)
        print(f"  D={d:>2}  {bar}  {c:>5} ({pct:5.1f}%)")

    # ── Phase composition ─────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  PHASE COMPOSITION  (mean tokens per sequence)")
    print(sep)
    total = agg['length_mean']
    for phase, key in [("Build ", 'build_mean'),
                       ("Undo  ", 'undo_mean'),
                       ("Tail  ", 'tail_mean')]:
        v = agg[key]
        pct = 100 * v / total if total > 0 else 0
        bar = _bar(v, total)
        print(f"  {phase}  {bar}  {v:5.1f} tok  ({pct:.1f}%)")

    # ── Stack depth stats ─────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  STACK DEPTH")
    print(sep)
    print(f"  Max stack depth  Mean ± Std : "
          f"{agg['max_stack_mean']:.2f} ± {agg['max_stack_std']:.2f}")

    # ── Token density ─────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  TOKEN DENSITY  (mean fraction of sequence)")
    print(sep)
    for label, key in [("u (undo) ", 'frac_u_mean'),
                       ("i (noise)", 'frac_i_mean')]:
        v = agg[key]
        bar = _bar(v, 1.0)
        print(f"  {label}  {bar}  {v:.4f}  ({100*v:.2f}%)")
    print(f"\n  [Training baseline: u ≈ 0.050, i ≈ 0.267 expected from P dist]")

    # ── Label balance ─────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  LABEL BALANCE  (ground-truth r-token values)")
    print(sep)
    tl = agg['total_labels']
    l0, l1 = agg['total_label_0'], agg['total_label_1']
    print(f"  Total r-labels : {tl:,}")
    print(f"  Label=0        : {l0:,}  ({100*l0/tl:.1f}%)" if tl else "  No labels")
    print(f"  Label=1        : {l1:,}  ({100*l1/tl:.1f}%)" if tl else "")
    print(f"  Balance ratio  : {l0/l1:.3f}" if l1 > 0 else "  (no label=1 found)")

    print(f"\n{'═'*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_datasets(paths: list[str], aggs: list[dict], labels: list[str],
                  out_path: str = "dataset_analysis.png"):
    if not HAS_MPL:
        print("[WARNING] matplotlib not available — skipping plots.")
        return

    n_datasets = len(paths)
    fig = plt.figure(figsize=(5 * n_datasets + 2, 14))
    fig.patch.set_facecolor('#0f1117')
    gs = gridspec.GridSpec(3, n_datasets, figure=fig,
                           hspace=0.45, wspace=0.35)

    COLORS = ['#4fc3f7', '#81c784', '#ffb74d', '#e57373', '#ce93d8']
    BG = '#1a1d27'
    GRID = '#2a2d3a'
    TEXT = '#e0e0e0'

    def ax_style(ax, title):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.set_title(title, color=TEXT, fontsize=10, pad=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.yaxis.label.set_color(TEXT)
        ax.xaxis.label.set_color(TEXT)
        ax.grid(axis='y', color=GRID, linewidth=0.5, linestyle='--')

    for col, (path, agg, label) in enumerate(zip(paths, aggs, labels)):
        color = COLORS[col % len(COLORS)]
        per_seq = agg['per_seq']

        # ── Row 0: Undo depth histogram ────────────────────────────────────
        ax0 = fig.add_subplot(gs[0, col])
        depths = [s['actual_undo_depth'] for s in per_seq]
        unique_d = sorted(set(depths))
        counts_d = [depths.count(d) for d in unique_d]
        ax0.bar(unique_d, counts_d, color=color, alpha=0.85, width=0.7)
        ax0.set_xlabel("Undo Depth D")
        ax0.set_ylabel("Count")
        ax_style(ax0, f"{label}\nUndo Depth Distribution")

        # ── Row 1: Phase composition (stacked bar per depth) ───────────────
        ax1 = fig.add_subplot(gs[1, col])
        if unique_d:
            build_m, undo_m, tail_m = [], [], []
            for d in unique_d:
                subset = [s for s in per_seq if s['actual_undo_depth'] == d]
                build_m.append(sum(s['build_tokens'] for s in subset) / len(subset))
                undo_m.append(sum(s['undo_tokens'] for s in subset) / len(subset))
                tail_m.append(sum(s['tail_tokens'] for s in subset) / len(subset))

            ax1.bar(unique_d, build_m, label='Build',
                    color='#4fc3f7', alpha=0.85, width=0.7)
            ax1.bar(unique_d, undo_m, bottom=build_m, label='Undo',
                    color='#e57373', alpha=0.85, width=0.7)
            ax1.bar(unique_d, tail_m,
                    bottom=[b + u for b, u in zip(build_m, undo_m)],
                    label='Tail', color='#81c784', alpha=0.85, width=0.7)
            ax1.legend(fontsize=8, facecolor=BG,
                       labelcolor=TEXT, framealpha=0.7)
        ax1.set_xlabel("Undo Depth D")
        ax1.set_ylabel("Mean Tokens")
        ax_style(ax1, "Phase Composition by Depth")

        # ── Row 2: u-token fraction vs depth ──────────────────────────────
        ax2 = fig.add_subplot(gs[2, col])
        if unique_d:
            frac_u_by_d = []
            for d in unique_d:
                subset = [s for s in per_seq if s['actual_undo_depth'] == d]
                frac_u_by_d.append(
                    sum(s['frac_u'] for s in subset) / len(subset)
                )
            ax2.plot(unique_d, frac_u_by_d, 'o-',
                     color=color, linewidth=2, markersize=5)
            ax2.axhline(0.05, color='#888', linestyle='--',
                        linewidth=1, label='Training P(u)=0.05')
            ax2.legend(fontsize=8, facecolor=BG,
                       labelcolor=TEXT, framealpha=0.7)
        ax2.set_xlabel("Undo Depth D")
        ax2.set_ylabel("Mean Fraction of u-tokens")
        ax_style(ax2, "u-Token Density vs Depth")

    fig.suptitle("UNDO Flip-Flop Dataset Analysis",
                 color=TEXT, fontsize=13, y=0.98)
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"[Plot saved → {out_path}]")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Analyse UNDO FF dataset statistics"
    )
    parser.add_argument('--input', nargs='+', required=True,
                        help="One or more .pt dataset files")
    parser.add_argument('--labels', nargs='+', default=None,
                        help="Display labels for each file (default: filename)")
    parser.add_argument('--plot', action='store_true',
                        help="Generate matplotlib summary figure")
    parser.add_argument('--plot_out', type=str, default='dataset_analysis.png',
                        help="Output path for the plot (default: dataset_analysis.png)")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.input):
        print("[ERROR] --labels count must match --input count", file=sys.stderr)
        sys.exit(1)

    labels = args.labels or [Path(p).stem for p in args.input]
    all_aggs = []

    for path, label in zip(args.input, labels):
        try:
            _, agg = analyse_dataset(path)
        except Exception as e:
            print(f"[ERROR] Could not load {path}: {e}", file=sys.stderr)
            sys.exit(1)
        all_aggs.append(agg)
        print_report(path, agg)

    if args.plot:
        if not HAS_MPL:
            print("[WARNING] Install matplotlib to enable plots: pip install matplotlib")
        else:
            plot_datasets(args.input, all_aggs, labels, args.plot_out)


if __name__ == '__main__':
    main()


# ── Example usage ──────────────────────────────────────────────────────────────
#
# Single file, text only:
#   python analyse_undo_dataset.py --input aggressive_sweep.pt
#
# Single file with plot:
#   python analyse_undo_dataset.py --input aggressive_sweep.pt --plot
#
# Compare two files side by side:
#   python analyse_undo_dataset.py \
#     --input aggressive_d5.pt aggressive_d10.pt \
#     --labels "D=5" "D=10" \
#     --plot --plot_out comparison.png
#
# Sweep across all per-depth files:
#   python analyse_undo_dataset.py \
#     --input aggressive_d1.pt aggressive_d3.pt aggressive_d5.pt \
#               aggressive_d7.pt aggressive_d10.pt aggressive_d15.pt \
#     --labels "D=1" "D=3" "D=5" "D=7" "D=10" "D=15" \
#     --plot --plot_out full_sweep.png