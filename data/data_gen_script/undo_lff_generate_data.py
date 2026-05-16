"""
This srcipt generates the [UNDO] Flip Flop language for training and testing. For standard Flip Flip generation, check other scripts
""" 

import argparse
import random
import torch

# ==========================================
# 1. Hardcoded Constants
# ==========================================
VOCAB = {'0': 0, '1': 1, 'r': 3, 'i': 4, 'w': 5, 'u': 6}
IGNORE_INDEX = -100

# The probability dist is the same with FFL(0.8)
P_I = 0.8
P_R = 0.1
P_W_BASE = 0.05
P_U_BASE = 0.05

def generate_single_sequence(target_length):
    """
    Generate single UNDO FF sequence and label
    containing dynamic probability adjustment and stack tracking
    """
    seq_tokens = []
    labels = []
    state_stack = []

    # Initialise the sequence with write
    v = str(random.choice(["0", "1"]))
    seq_tokens.extend(['w', v])
    labels.extend([IGNORE_INDEX, IGNORE_INDEX])
    state_stack.append(v)
    
    current_len = 2
    
    # Generate the mid sequence
    while current_len < target_length - 1:
        # Calculate the number of tokens remaining in the loop body (reserving one position for the mandatory trailing “r”
        remaining = (target_length - 1) - current_len

        if remaining == 1:
        # Only single-token instructions can be inserted; confirm the instruction directly here and skip the general sampling.
            if len(state_stack) > 1:
                ins = random.choices(['r', 'u'], weights=[P_R, P_U_BASE], k=1)[0]
            else:
                ins = 'r'
        else:
            if len(state_stack) > 1:
                choices = ['i', 'r', 'w', 'u']
                weights = [P_I, P_R, P_W_BASE, P_U_BASE]
            else:
                choices = ['i', 'r', 'w']
                # Compensate for the 0.05 probability of [UNDO] by adjusting the write rate to maintain the overall update rate, or redistribute it proportionally
                weights = [P_I, P_R, P_W_BASE + P_U_BASE]  
            ins = random.choices(choices, weights=weights, k=1)[0]
        
        if ins == 'w':
            val = str(random.choice(["0", "1"]))
            seq_tokens.extend(['w', val])
            labels.extend([IGNORE_INDEX, IGNORE_INDEX])
            state_stack.append(val)
            current_len += 2
        elif ins == 'u':
            seq_tokens.append('u')
            labels.append(IGNORE_INDEX)
            state_stack.pop() # Deterministic rollback
            current_len += 1
        elif ins == 'i':
            val = str(random.choice(["0", "1"]))
            seq_tokens.extend(['i', val])
            labels.extend([IGNORE_INDEX, IGNORE_INDEX])
            current_len += 2
        elif ins == 'r':
            seq_tokens.append('r')
            labels.append(int(state_stack[-1])) # Only those followed by an r are accompanied by a truth value label
            current_len += 1

    # Force the sequence to end with “r” (if it does not end with “r”, append one)
    if seq_tokens[-1] != 'r':
        seq_tokens.append('r')
        labels.append(int(state_stack[-1]))

    # Convert to Tensor
    seq_ids = torch.tensor([VOCAB[t] if t in VOCAB else int(t) for t in seq_tokens], dtype=torch.long)
    label_ids = torch.tensor(labels, dtype=torch.long)
    
    return seq_ids, label_ids

def main(args):
    dataset = []
    for _ in range(args.num_samples):
        # Select a target length at random from within the specified range
        target_len = random.randint(args.min_len, args.max_len)
        seq, label = generate_single_sequence(target_len)
        dataset.append({'input': seq, 'label': label})
        
    torch.save(dataset, args.output_file)
    print(f"Saved {args.num_samples} sequences to {args.output_file} (Len: {args.min_len}-{args.max_len})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate UNDO Flip-Flop Dataset")
    parser.add_argument('--num_samples', type=int, required=True)
    parser.add_argument('--min_len', type=int, required=True)
    parser.add_argument('--max_len', type=int, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()
    main(args)


#1. Generate train set(Train ID)**
#python undo_lff_generate_data.py --num_samples 10000 --min_len 1 --max_len 50 --output_file undo_train_id.pt


#2. Generate in-dist test set (Test ID)
#python undo_lff_generate_data.py --num_samples 2000 --min_len 1 --max_len 50 --output_file undo_test_id.pt

#3. Generate out of dist test set (Test OOD)**
#python undo_lff_generate_data.py --num_samples 2000 --min_len 51 --max_len 100 --output_file undo_test_ood.pt

