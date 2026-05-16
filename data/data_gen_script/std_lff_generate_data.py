"""
This script generates the standard Flip-Flop language for training and testing.
Serves as the baseline counterpart to the UNDO Flip-Flop variant.

Probability design (mirrors UNDO-FF total state-change rate):
  P_I = 0.80  (ignore)
  P_R = 0.10  (read)
  P_W = 0.10  (write) ← equivalent to P_W_BASE + P_U_BASE in UNDO-FF
  [UNDO] is absent; 'u' is retained in VOCAB for embedding parity.
"""

import argparse
import random
import torch

# ==========================================
# 1. Hardcoded Constants
# ==========================================
# Identical to UNDO-FF: 'u' retained for vocab/embedding parity
VOCAB = {'0': 0, '1': 1, 'r': 3, 'i': 4, 'w': 5, 'u': 6}
IGNORE_INDEX = -100

P_I = 0.80
P_R = 0.10
P_W = 0.10  # = P_W_BASE + P_U_BASE in UNDO-FF; no UNDO branch here

def generate_single_sequence(target_length):
    """
    Generate a single standard Flip-Flop sequence and its labels.
    State is tracked as a single scalar (no stack needed without UNDO).
    """
    seq_tokens = []
    labels = []

    # Initialise with a write
    v = str(random.choice(["0", "1"]))
    seq_tokens.extend(['w', v])
    labels.extend([IGNORE_INDEX, IGNORE_INDEX])
    current_state = v
    current_len = 2

    # Generate the mid sequence
    while current_len < target_length - 1:
        # Guard: check remaining space before sampling
        remaining = (target_length - 1) - current_len

        if remaining == 1:
            # Only single-token instructions fit; 'r' is the only sensible choice
            # (no UNDO here, so no 'u' fallback needed)
            ins = 'r'
        else:
            ins = random.choices(
                ['i', 'r', 'w'],
                weights=[P_I, P_R, P_W],
                k=1
            )[0]

        if ins == 'w':
            val = str(random.choice(["0", "1"]))
            seq_tokens.extend(['w', val])
            labels.extend([IGNORE_INDEX, IGNORE_INDEX])
            current_state = val
            current_len += 2
        elif ins == 'i':
            val = str(random.choice(["0", "1"]))
            seq_tokens.extend(['i', val])
            labels.extend([IGNORE_INDEX, IGNORE_INDEX])
            current_len += 2
        elif ins == 'r':
            seq_tokens.append('r')
            labels.append(int(current_state))
            current_len += 1

    # Force sequence to end with 'r'
    if seq_tokens[-1] != 'r':
        seq_tokens.append('r')
        labels.append(int(current_state))

    # Convert to Tensor
    seq_ids = torch.tensor(
        [VOCAB[t] if t in VOCAB else int(t) for t in seq_tokens],
        dtype=torch.long
    )
    label_ids = torch.tensor(labels, dtype=torch.long)

    return seq_ids, label_ids

def main(args):
    dataset = []
    for _ in range(args.num_samples):
        target_len = random.randint(args.min_len, args.max_len)
        seq, label = generate_single_sequence(target_len)
        dataset.append({'input': seq, 'label': label})

    torch.save(dataset, args.output_file)
    print(f"Saved {args.num_samples} sequences to {args.output_file} (Len: {args.min_len}-{args.max_len})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Standard Flip-Flop Dataset")
    parser.add_argument('--num_samples', type=int, required=True)
    parser.add_argument('--min_len',     type=int, required=True)
    parser.add_argument('--max_len',     type=int, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()
    main(args)

# 1. Train set (ID)
# python std_lff_generate_data.py --num_samples 10000 --min_len 1 --max_len 50  --output_file std_train_id_ff.pt

# 2. In-distribution test set
# python std_lff_generate_data.py --num_samples 2000  --min_len 1 --max_len 50  --output_file std_test_id_ff.pt

# 3. Out-of-distribution test set
# python std_lff_generate_data.py --num_samples 2000  --min_len 51 --max_len 100 --output_file std_test_ood_ff.pt