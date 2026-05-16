import torch

data = torch.load('aggressive_d10.pt')

# Overall structure
print(type(data))          # <class 'list'>
print(len(data))           # 10000
print(data[0].keys())      # dict_keys(['input', 'label'])

# First Sample 
sample = data[0]
print(sample['input'])     # tensor([5, 0, 4, 1, 3, ...])
print(sample['label'])     # tensor([-100, -100, -100, -100, 0, ...])
print(sample['input'].dtype)   # torch.int64
print(sample['input'].shape)   # torch.Size([N])  ← the actual length of the sequence
print(sample['label'].shape)   # torch.Size([N])  ← the same length with the input

# All the seq length dist
lengths = [d['input'].shape[0] for d in data]
print(f"min len: {min(lengths)}, max len: {max(lengths)}, mean: {sum(lengths)/len(lengths):.1f}")