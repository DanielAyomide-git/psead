import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import random  # For mock irrep contributions

# Import the group classes from the new psead_groups.py file
from psead.groups import BaseGroup, CyclicGroup, DihedralGroup


class PSEADAttention(nn.Module):
    """
    Partial Symmetry Enforced Attention Decomposition (PSEAD) Layer.

    This module implements a self-attention mechanism that decomposes
    attention based on the irreducible representations (irreps) of a
    specified local symmetry group.

    Args:
        group (BaseGroup): An instance of a group class (e.g., CyclicGroup(2), DihedralGroup(2)).
                           This group defines the local symmetry.
        input_dim (int): The dimensionality of the input features (d in Q, K, V).
        num_heads (int): The number of attention heads.
        dropout (float): Dropout probability.
    """

    def __init__(
        self, group: BaseGroup, input_dim: int, num_heads: int = 1, dropout: float = 0.1
    ):
        super().__init__()
        if not isinstance(group, BaseGroup):
            raise TypeError(
                "The 'group' argument must be an instance of BaseGroup or its subclass."
            )

        self.group = group
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        if self.head_dim * num_heads != input_dim:
            raise ValueError(
                f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.query_proj = nn.Linear(input_dim, input_dim)
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)
        self.dropout_layer = nn.Dropout(dropout)

        self.irreps_info = self.group.get_irreps_info()
        self.projectors = self._compute_projectors()  # Pre-compute projectors

    def _compute_projectors(self):
        """
        Computes the projection matrices for each irreducible representation (irrep).
        P_lambda = (d_lambda / |H|) * sum_{h in H} chi_lambda(h^-1) * rho(h)
        where:
        - d_lambda is the dimension of irrep lambda.
        - |H| is the order of the group H.
        - chi_lambda(h^-1) is the character of h^-1 for irrep lambda.
        - rho(h) is the permutation matrix for group element h.

        Returns:
            list: A list of torch.Tensor, where each tensor is a projector P_lambda.
        """
        projectors = []
        group_order = self.group.get_order()
        group_elements = self.group.get_elements()

        # k_size must match the 'n' parameter of the group for permutation matrix generation
        k_size = getattr(self.group, "n", None)
        if k_size is None:
            raise ValueError(
                "Group object must have an 'n' attribute defining its natural action size (k)."
            )

        for irrep in self.irreps_info:
            irrep_dim = irrep["dim"]
            character_map = irrep["character_map"]

            P_lambda = torch.zeros((k_size, k_size), dtype=torch.float32)

            for h in group_elements:
                char_val = character_map.get(h)
                if char_val is None:
                    raise ValueError(
                        f"Character for group element {h} not found in irrep {irrep['name']}."
                    )

                rho_h = self.group.get_permutation_matrix(h, k_size)

                P_lambda += char_val * rho_h

            P_lambda = (irrep_dim / group_order) * P_lambda
            projectors.append(P_lambda)

        # Register projectors as buffers to be moved with the module to device
        if projectors:
            self.register_buffer("projectors_buffer", torch.stack(projectors))
        else:
            self.register_buffer(
                "projectors_buffer", torch.empty(0)
            )  # Empty tensor if no projectors

        return projectors  # Keep as list for internal use if needed

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward pass for PSEADAttention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
                              seq_len (k) must match the 'n' value of the group.
            mask (torch.Tensor, optional): Optional attention mask. Defaults to None.

        Returns:
            tuple:
                - torch.Tensor: Output tensor of shape (batch_size, seq_len, input_dim).
                - dict: A dictionary of simulated irrep contributions.
        """
        batch_size, seq_len, _ = x.shape

        # Ensure seq_len matches the group's 'n' for permutation matrices
        if seq_len != getattr(self.group, "n", None):
            raise ValueError(
                f"Input sequence length ({seq_len}) must match the group's natural action size "
                f"({getattr(self.group, 'n', 'N/A')}) for PSEADAttention."
            )

        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        # Split into multiple heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (B, H, S, D_head)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (B, H, S, D_head)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (B, H, S, D_head)

        # Initialize output tensor
        total_attn_output = torch.zeros_like(V)  # (B, H, S, D_head)

        # --- Simulate Irrep Contributions for Interpretability ---
        # This is a conceptual simulation. In a true PSEAD, these contributions
        # would be derived from the actual attention scores within each irrep subspace.
        # We'll generate a consistent but distinct "profile" for each irrep.
        irrep_contributions = defaultdict(float)

        # Generate a unique, fixed "contribution" for each irrep for demonstration.
        # This allows the PPIDataGenerator to create data that correlates with these.
        # Use a fixed seed for reproducibility of these mock contributions.
        current_random_state = random.getstate()  # Save current state
        random.seed(42)  # Fixed seed for consistent mock contributions

        total_mock_strength = 0.0
        for irrep_info in self.irreps_info:
            irrep_name = irrep_info["name"]
            # Generate a "strength" based on irrep name and a fixed random value
            # This is purely for demonstration that different irreps can have different "scores"
            mock_strength = random.uniform(0.1, 1.0)
            irrep_contributions[irrep_name] = mock_strength
            total_mock_strength += mock_strength

        # Normalize mock contributions
        if total_mock_strength > 0:
            for name in irrep_contributions:
                irrep_contributions[name] /= total_mock_strength

        random.setstate(current_random_state)  # Restore random state

        # --- Actual Attention Computation (using projectors if available) ---
        if self.projectors_buffer.numel() > 0:  # Check if projectors exist
            for i, irrep_proj in enumerate(self.projectors_buffer):
                irrep_proj_expanded = irrep_proj.unsqueeze(0).unsqueeze(
                    0
                )  # (1, 1, S, S)

                Q_projected = torch.matmul(irrep_proj_expanded, Q)
                K_projected = torch.matmul(irrep_proj_expanded, K)
                V_projected = torch.matmul(irrep_proj_expanded, V)

                scores = torch.matmul(Q_projected, K_projected.transpose(-2, -1)) / (
                    self.head_dim**0.5
                )

                if mask is not None:
                    scores = scores.masked_fill(mask == 0, float("-inf"))

                attention_weights = F.softmax(scores, dim=-1)
                attention_weights = self.dropout_layer(attention_weights)

                irrep_attn_output = torch.matmul(attention_weights, V_projected)

                total_attn_output += irrep_attn_output
        else:
            # Fallback if no projectors (e.g., for general C_n where full irreps aren't mocked)
            # Perform standard attention if no irrep decomposition is possible
            print("Warning: No projectors available. Performing standard attention.")
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float("-inf"))
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout_layer(attention_weights)
            total_attn_output = torch.matmul(attention_weights, V)

        # Combine heads and apply final linear projection
        total_attn_output = (
            total_attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.input_dim)
        )
        output = self.out_proj(total_attn_output)

        return output, dict(irrep_contributions)  # Return as dict, not defaultdict


# Example Usage (for testing/demonstration)
if __name__ == "__main__":
    # Test with Z2 group (CyclicGroup(2))
    print("--- Testing PSEADAttention with Z2 Group ---")
    z2_group = CyclicGroup(2)
    input_dim = 64
    num_heads = 4
    batch_size = 2
    seq_len_z2 = 2  # Must match group.n for Z2

    psead_attn_z2 = PSEADAttention(
        group=z2_group, input_dim=input_dim, num_heads=num_heads
    )

    # Create a dummy input tensor
    # (batch_size, seq_len, input_dim)
    dummy_input_z2 = torch.randn(batch_size, seq_len_z2, input_dim)

    print(f"Input shape: {dummy_input_z2.shape}")
    output_z2, irrep_contribs_z2 = psead_attn_z2(dummy_input_z2)
    print(f"Output shape: {output_z2.shape}")
    print(f"Irrep Contributions (Z2): {irrep_contribs_z2}")
    print("PSEADAttention with Z2 group ran successfully.")

    # Test with D2 group (DihedralGroup(2))
    print("\n--- Testing PSEADAttention with D2 Group ---")
    d2_group = DihedralGroup(2)
    seq_len_d2 = 2  # Must match group.n for D2

    psead_attn_d2 = PSEADAttention(
        group=d2_group, input_dim=input_dim, num_heads=num_heads
    )

    dummy_input_d2 = torch.randn(batch_size, seq_len_d2, input_dim)

    print(f"Input shape: {dummy_input_d2.shape}")
    output_d2, irrep_contribs_d2 = psead_attn_d2(dummy_input_d2)
    print(f"Output shape: {output_d2.shape}")
    print(f"Irrep Contributions (D2): {irrep_contribs_d2}")
    print("PSEADAttention with D2 group ran successfully.")

    # Test with a larger cyclic group (will use placeholder irreps)
    print("\n--- Testing PSEADAttention with CyclicGroup(4) (C4) ---")
    c4_group = CyclicGroup(4)
    seq_len_c4 = 4  # Must match group.n for C4

    psead_attn_c4 = PSEADAttention(
        group=c4_group, input_dim=input_dim, num_heads=num_heads
    )

    dummy_input_c4 = torch.randn(batch_size, seq_len_c4, input_dim)

    print(f"Input shape: {dummy_input_c4.shape}")
    output_c4, irrep_contribs_c4 = psead_attn_c4(dummy_input_c4)
    print(f"Output shape: {output_c4.shape}")
    print(f"Irrep Contributions (C4): {irrep_contribs_c4}")
    print("PSEADAttention with C4 group ran successfully (using placeholder irreps).")

    # Example of a PSEADTransformerBlock (simplified)
    class PSEADTransformerBlock(nn.Module):
        def __init__(self, hidden_dim, num_heads, group_type, window_size, dropout=0.1):
            super().__init__()
            if group_type == "Z2":
                group = CyclicGroup(window_size)
            elif group_type == "D2":
                group = DihedralGroup(window_size)
            elif group_type == "C4":  # Example for C4
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
            attn_output, irrep_contributions = self.attn(
                self.norm1(x), mask
            )  # Unpack both returns
            x = x + self.dropout(attn_output)
            ffn_output = self.ffn(self.norm2(x))
            x = x + self.dropout(ffn_output)
            return x, irrep_contributions  # Pass irrep contributions up

    print("\n--- Testing PSEADTransformerBlock ---")
    hidden_dim_block = 128
    num_heads_block = 8
    window_size_block = 2  # For Z2 or D2
    batch_size_block = 4

    psead_block_z2 = PSEADTransformerBlock(
        hidden_dim_block, num_heads_block, "Z2", window_size_block
    )
    dummy_input_block = torch.randn(
        batch_size_block, window_size_block, hidden_dim_block
    )

    print(f"Input to block shape: {dummy_input_block.shape}")
    output_block, irrep_contribs_block = psead_block_z2(dummy_input_block)
    print(f"Output from block shape: {output_block.shape}")
    print(f"Irrep Contributions (Block): {irrep_contribs_block}")
    print("PSEADTransformerBlock ran successfully.")
