import torch
import torch.nn as nn
from psead.attention import PSEADAttention  # Assuming psead library integration
from psead.groups import CyclicGroup, DihedralGroup


class PSEADTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, group_type, window_size, dropout=0.1):
        super().__init__()
        if group_type == "Z2":
            group = CyclicGroup(window_size)
        elif group_type == "D2":
            group = DihedralGroup(window_size)
        # Add more group types as needed, e.g., for C4, D4 etc.
        elif group_type == "C4":
            group = CyclicGroup(window_size)
        else:
            raise ValueError(f"Unsupported group type: {group_type}")

        self.attn = PSEADAttention(group, hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attn(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_output)
        return x


class PSEADPolicyNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_heads,
        num_layers,
        group_type,
        window_size,
        action_dim,
    ):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.psead_layers = nn.ModuleList(
            [
                PSEADTransformerBlock(hidden_dim, num_heads, group_type, window_size)
                for _ in range(num_layers)
            ]
        )
        self.output_layer = nn.Linear(
            hidden_dim, action_dim
        )  # action_dim depends on action representation

    def forward(self, observation):
        # observation could be a batch of contact map patches or residue features
        # Assuming observation is (batch_size, window_size, feature_dim)
        embedded_obs = self.embedding(observation)

        for layer in self.psead_layers:
            embedded_obs = layer(
                embedded_obs
            )  # PSEADTransformerBlock applies attention decomposition

        action_logits = self.output_layer(
            embedded_obs.mean(dim=1)
        )  # Pool features to get action logits
        return action_logits


# --- Test Cases ---

print("--- Testing PSEADPolicyNetwork with D2 group ---")
input_feature_dim_d2 = 64
hidden_dim_d2 = 128
num_heads_d2 = 8
num_layers_d2 = 4
action_dim_d2 = 10
window_size_d2 = 2  # For D2, the natural action is on 2 elements

policy_network_d2 = PSEADPolicyNetwork(
    input_feature_dim_d2,
    hidden_dim_d2,
    num_heads_d2,
    num_layers_d2,
    group_type="D2",
    window_size=window_size_d2,
    action_dim=action_dim_d2,
)

# Create a dummy observation for the D2 network
# Shape: (batch_size, window_size, input_feature_dim)
dummy_observation_d2 = torch.randn(1, window_size_d2, input_feature_dim_d2)

print(f"\nDummy Observation (D2) shape: {dummy_observation_d2.shape}")
output_logits_d2 = policy_network_d2(dummy_observation_d2)
print(f"Output Logits (D2) shape: {output_logits_d2.shape}")

# Expected output shape: (batch_size, action_dim)
assert output_logits_d2.shape == (1, action_dim_d2), "D2 network output shape mismatch!"
print("D2 network test successful: Output shape matches expected.")


print("\n--- Testing PSEADPolicyNetwork with Z2 group ---")
input_feature_dim_z2 = 64
hidden_dim_z2 = 128
num_heads_z2 = 8
num_layers_z2 = 4
action_dim_z2 = 5
window_size_z2 = 4  # Z2 (reflection) can act on any window size, e.g., 4

policy_network_z2 = PSEADPolicyNetwork(
    input_feature_dim_z2,
    hidden_dim_z2,
    num_heads_z2,
    num_layers_z2,
    group_type="Z2",
    window_size=window_size_z2,
    action_dim=action_dim_z2,
)

# Create a dummy observation for the Z2 network
dummy_observation_z2 = torch.randn(2, window_size_z2, input_feature_dim_z2)

print(f"\nDummy Observation (Z2) shape: {dummy_observation_z2.shape}")
output_logits_z2 = policy_network_z2(dummy_observation_z2)
print(f"Output Logits (Z2) shape: {output_logits_z2.shape}")

# Expected output shape: (batch_size, action_dim)
assert output_logits_z2.shape == (2, action_dim_z2), "Z2 network output shape mismatch!"
print("Z2 network test successful: Output shape matches expected.")


print("\n--- Testing PSEADPolicyNetwork with C4 group ---")
input_feature_dim_c4 = 64
hidden_dim_c4 = 128
num_heads_c4 = 8
num_layers_c4 = 4
action_dim_c4 = 7
window_size_c4 = 4  # For C4, the natural action is on 4 elements

policy_network_c4 = PSEADPolicyNetwork(
    input_feature_dim_c4,
    hidden_dim_c4,
    num_heads_c4,
    num_layers_c4,
    group_type="C4",
    window_size=window_size_c4,
    action_dim=action_dim_c4,
)

# Create a dummy observation for the C4 network
dummy_observation_c4 = torch.randn(3, window_size_c4, input_feature_dim_c4)

print(f"\nDummy Observation (C4) shape: {dummy_observation_c4.shape}")
output_logits_c4 = policy_network_c4(dummy_observation_c4)
print(f"Output Logits (C4) shape: {output_logits_c4.shape}")

# Expected output shape: (batch_size, action_dim)
assert output_logits_c4.shape == (3, action_dim_c4), "C4 network output shape mismatch!"
print("C4 network test successful: Output shape matches expected.")

print("\nAll basic network instantiation and forward pass tests completed.")
