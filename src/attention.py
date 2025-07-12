import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Assuming psead.groups is in the same directory or accessible via PYTHONPATH
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
        num_heads (int): The number of attention heads. For simplicity, this implementation
                         distributes the input_dim across heads.
        dropout (float): Dropout probability.
    """
    def __init__(self, group: BaseGroup, input_dim: int, num_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        if not isinstance(group, BaseGroup):
            raise TypeError("The 'group' argument must be an instance of BaseGroup or its subclass.")

        self.group = group
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        if self.head_dim * num_heads != input_dim:
            raise ValueError(f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads})")

        self.query_proj = nn.Linear(input_dim, input_dim)
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

        self.irreps_info = self.group.get_irreps_info()
        self.projectors = self._compute_projectors() # Pre-compute projectors

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

        # Determine k (window size) from the expected input shape during forward pass
        # This is a bit tricky during __init__ as k is not known.
        # We will assume k will be the sequence length of the input.
        # The permutation matrices must be k x k.
        # For simplicity, we'll compute projectors for a generic k and assume
        # the input sequence length matches this k during forward.
        # A more robust solution might dynamically compute projectors or
        # require k during init. For now, let's assume k is implicitly known
        # from the group's natural action (e.g., k=n for C_n, D_n).
        # We'll use the group's 'n' parameter as k for permutation matrix generation.
        # This implies the input window size must match self.group.n.
        k_size = getattr(self.group, 'n', None)
        if k_size is None:
            raise ValueError("Group object must have an 'n' attribute defining its natural action size (k).")

        for irrep in self.irreps_info:
            irrep_dim = irrep['dim']
            character_map = irrep['character_map']
            
            # Initialize projector for this irrep
            P_lambda = torch.zeros((k_size, k_size), dtype=torch.float32)

            for h in group_elements:
                # Find h_inv (h inverse)
                # For permutation groups, h_inv is often related to the inverse permutation.
                # For C_n, h_inv is (n - h) % n.
                # For D_n, finding inverse is more complex.
                # A simpler approach is to iterate over all elements and their characters.
                # If the character map stores chi_lambda(h) directly, we need chi_lambda(h_inv).
                # For real irreps, chi_lambda(h_inv) = chi_lambda(h).
                # For complex irreps, chi_lambda(h_inv) = conj(chi_lambda(h)).
                # Given our current simplified character maps (real values), we'll assume chi_lambda(h_inv) = chi_lambda(h).
                # This holds for real representations.

                # Get the character value for h (assuming character_map directly gives chi(h))
                char_val = character_map.get(h)
                if char_val is None:
                    raise ValueError(f"Character for group element {h} not found in irrep {irrep['name']}.")

                # Get permutation matrix for h
                rho_h = self.group.get_permutation_matrix(h, k_size)
                
                # Accumulate for the projector sum
                P_lambda += char_val * rho_h

            # Normalize and scale by irrep dimension
            P_lambda = (irrep_dim / group_order) * P_lambda
            projectors.append(P_lambda)
        
        # Move projectors to the correct device when the module is moved
        self.register_buffer('projectors_buffer', torch.stack(projectors)) # Stack into a single tensor

        return projectors # Keep as list for easier iteration if needed, but buffer is main storage

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward pass for PSEADAttention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
                              seq_len (k) must match the 'n' value of the group.
            mask (torch.Tensor, optional): Optional attention mask. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, input_dim).
        """
        batch_size, seq_len, _ = x.shape

        # Ensure seq_len matches the group's 'n' for permutation matrices
        if seq_len != getattr(self.group, 'n', None):
            raise ValueError(
                f"Input sequence length ({seq_len}) must match the group's natural action size "
                f"({getattr(self.group, 'n', 'N/A')}) for PSEADAttention."
            )

        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        # Split into multiple heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, S, D_head)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, S, D_head)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, S, D_head)

        # Initialize output tensor
        total_attn_output = torch.zeros_like(V) # (B, H, S, D_head)

        # Iterate through pre-computed projectors for each irrep
        for i, irrep_proj in enumerate(self.projectors_buffer):
            # Apply projector to Q, K, V for each head
            # irrep_proj shape: (k, k)
            # Q, K, V shape: (B, H, S, D_head)
            # We need to apply P_lambda to the 'S' dimension (sequence length)
            # P_lambda @ Q_s, where Q_s is (S, D_head) for a single head/batch
            
            # Unsqueeze projector to (1, 1, k, k) for broadcasting across batch and heads
            irrep_proj_expanded = irrep_proj.unsqueeze(0).unsqueeze(0) # (1, 1, S, S)

            Q_projected = torch.matmul(irrep_proj_expanded, Q) # (B, H, S, D_head)
            K_projected = torch.matmul(irrep_proj_expanded, K) # (B, H, S, D_head)
            V_projected = torch.matmul(irrep_proj_expanded, V) # (B, H, S, D_head)

            # Compute scaled dot-product attention for this irrep
            scores = torch.matmul(Q_projected, K_projected.transpose(-2, -1)) / (self.head_dim ** 0.5)

            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))

            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)

            irrep_attn_output = torch.matmul(attention_weights, V_projected) # (B, H, S, D_head)
            
            # Accumulate the irrep-specific attention outputs
            total_attn_output += irrep_attn_output
        
        # Combine heads and apply final linear projection
        total_attn_output = total_attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.input_dim)
        output = self.out_proj(total_attn_output)
        
        return output

# Example Usage (for testing/demonstration)
if __name__ == '__main__':
    # Test with Z2 group (CyclicGroup(2))
    print("--- Testing PSEADAttention with Z2 Group ---")
    z2_group = CyclicGroup(2)
    input_dim = 64
    num_heads = 4
    batch_size = 2
    seq_len_z2 = 2 # Must match group.n for Z2

    psead_attn_z2 = PSEADAttention(group=z2_group, input_dim=input_dim, num_heads=num_heads)
    
    # Create a dummy input tensor
    # (batch_size, seq_len, input_dim)
    dummy_input_z2 = torch.randn(batch_size, seq_len_z2, input_dim)
    
    print(f"Input shape: {dummy_input_z2.shape}")
    output_z2 = psead_attn_z2(dummy_input_z2)
    print(f"Output shape: {output_z2.shape}")
    print("PSEADAttention with Z2 group ran successfully.")

    # Test with D2 group (DihedralGroup(2))
    print("\n--- Testing PSEADAttention with D2 Group ---")
    d2_group = DihedralGroup(2)
    seq_len_d2 = 2 # Must match group.n for D2

    psead_attn_d2 = PSEADAttention(group=d2_group, input_dim=input_dim, num_heads=num_heads)
    
    dummy_input_d2 = torch.randn(batch_size, seq_len_d2, input_dim)
    
    print(f"Input shape: {dummy_input_d2.shape}")
    output_d2 = psead_attn_d2(dummy_input_d2)
    print(f"Output shape: {output_d2.shape}")
    print("PSEADAttention with D2 group ran successfully.")

    # Test with a larger cyclic group (will use placeholder irreps)
    print("\n--- Testing PSEADAttention with CyclicGroup(4) (C4) ---")
    c4_group = CyclicGroup(4)
    seq_len_c4 = 4 # Must match group.n for C4
    
    psead_attn_c4 = PSEADAttention(group=c4_group, input_dim=input_dim, num_heads=num_heads)
    
    dummy_input_c4 = torch.randn(batch_size, seq_len_c4, input_dim)
    
    print(f"Input shape: {dummy_input_c4.shape}")
    output_c4 = psead_attn_c4(dummy_input_c4)
    print(f"Output shape: {output_c4.shape}")
    print("PSEADAttention with C4 group ran successfully (using placeholder irreps).")

    # Example of a PSEADTransformerBlock (simplified)
    class PSEADTransformerBlock(nn.Module):
        def __init__(self, hidden_dim, num_heads, group_type, window_size, dropout=0.1):
            super().__init__()
            if group_type == "Z2":
                group = CyclicGroup(window_size)
            elif group_type == "D2":
                group = DihedralGroup(window_size)
            elif group_type == "C4": # Example for C4
                group = CyclicGroup(window_size)
            else:
                raise ValueError(f"Unsupported group type: {group_type}")

            self.attn = PSEADAttention(group, hidden_dim, num_heads, dropout)
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.ReLU(),
                nn.Linear(4 * hidden_dim, hidden_dim),
                nn.Dropout(dropout)
            )
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask=None):
            attn_output = self.attn(self.norm1(x), mask)
            x = x + self.dropout(attn_output)
            ffn_output = self.ffn(self.norm2(x))
            x = x + self.dropout(ffn_output)
            return x

    print("\n--- Testing PSEADTransformerBlock ---")
    hidden_dim_block = 128
    num_heads_block = 8
    window_size_block = 2 # For Z2 or D2
    batch_size_block = 4

    psead_block_z2 = PSEADTransformerBlock(hidden_dim_block, num_heads_block, "Z2", window_size_block)
    dummy_input_block = torch.randn(batch_size_block, window_size_block, hidden_dim_block)
    
    print(f"Input to block shape: {dummy_input_block.shape}")
    output_block = psead_block_z2(dummy_input_block)
    print(f"Output from block shape: {output_block.shape}")
    print("PSEADTransformerBlock ran successfully.")

