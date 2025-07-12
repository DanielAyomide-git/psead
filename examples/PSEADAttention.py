import torch
from psead.attention import PSEADAttention
from psead.groups import CyclicGroup

# Example with Z2 group (CyclicGroup(2))
z2_group = CyclicGroup(2)
input_dim = 64
num_heads = 4
batch_size = 2
seq_len_z2 = 2  # Must match group.n for Z2

psead_attn_z2 = PSEADAttention(group=z2_group, input_dim=input_dim, num_heads=num_heads)

dummy_input_z2 = torch.randn(batch_size, seq_len_z2, input_dim)

print(f"Input shape: {dummy_input_z2.shape}")
output_z2 = psead_attn_z2(dummy_input_z2)
print(f"Output shape: {output_z2.shape}")
