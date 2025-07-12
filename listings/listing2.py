import torch
import torch.nn as nn
from psead.attention import PSEADAttention
from psead.groups import CyclicGroup


# Example: Encode a DNA window (e.g., 10-mer) into numerical features
def encode_dna_window(dna_sequence: str, char_to_vec_map: dict):
    # Example: one-hot encoding or learned embeddings
    # 'A': [1,0,0,0], 'C': [0,1,0,0], ...
    encoded_seq = torch.tensor(
        [char_to_vec_map[base] for base in dna_sequence], dtype=torch.float32
    )
    return encoded_seq.unsqueeze(0)  # Add batch dimension


# Assume a simple character to vector map
char_to_vec = {
    "A": [1.0, 0, 0, 0],
    "C": [0, 1.0, 0, 0],
    "G": [0, 0, 1.0, 0],
    "T": [0, 0, 0, 1.0],
}
feature_dim = 4  # One-hot encoding dimension

# Example DNA window with Z2 symmetry (palindrome)
dna_window_seq_2mer = "AG"  # k=2 for Z2 example
dna_window_features = encode_dna_window(
    dna_window_seq_2mer, char_to_vec
)  # shape (1, 2, 4)

# Define query, key, value projections (simplified for illustration)
# In a real Transformer, these would be nn.Linear layers
Q = K = V = dna_window_features

# Instantiate PSEADAttention for Z2 group
psead_attn_layer = PSEADAttention(group=CyclicGroup(2), input_dim=feature_dim)

# Compute the PSEAD-augmented attention output
out_psead_attn = psead_attn_layer(Q, K, V)

print(f"Input shape: {Q.shape}")
print(f"Output shape from PSEAD attention: {out_psead_attn.shape}")
