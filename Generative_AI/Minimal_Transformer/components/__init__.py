import random

def generate_copy_data(num_samples=1000, seq_len=10, vocab_size=20):
    data = []
    for _ in range(num_samples):
        seq = [random.randint(2, vocab_size-1) for _ in range(seq_len)]
        data.append((seq, seq))
    return data
