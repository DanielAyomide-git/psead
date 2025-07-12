import torch
import torch.nn as nn
from psead.attention import PSEADAttention
from psead.groups import DihedralGroup, CyclicGroup

# Example: Protein residue features for a D2-symmetric motif (k=2)
# For a general D_n, k would be n. This example uses D2 with k=2.
window_size_protein = 2  # Number of 'elements' in our D2-symmetric motif
feature_dim_protein = 128  # Feature dimension for each element
protein_features = torch.randn(
    1, window_size_protein, feature_dim_protein
)  # (batch_size, k, feature_dim)

# Define Q, K, V (simplified)
Q_p = K_p = V_p = protein_features

# Instantiate PSEADAttention for Dihedral Group D2
psead_attn_d2 = PSEADAttention(group=DihedralGroup(2), input_dim=feature_dim_protein)

# Compute the PSEAD-augmented attention output
out_psead_attn_d2 = psead_attn_d2(Q_p, K_p, V_p)

print(f"Input protein feature shape: {Q_p.shape}")
print(
    f"Output protein feature shape from PSEAD attention (D2): {out_psead_attn_d2.shape}"
)

# For a cyclic C3 symmetry in a coiled-coil (requires k=3):
# psead_attn_c3 = PSEADAttention(group=CyclicGroup(3), input_dim=feature_dim_protein)
# # dummy_input_c3 = torch.randn(1, 3, feature_dim_protein)
# # out_psead_attn_c3 = psead_attn_c3(dummy_input_c3, dummy_input_c3, dummy_input_c3)
# print("Note: CyclicGroup(3) uses placeholder irreps in current PSEAD implementation.")
