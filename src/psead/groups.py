import numpy as np
import torch

class BaseGroup:
    """
    Base class for defining a finite group.
    Subclasses must implement:
    - __init__: To define group elements and order.
    - get_elements(): Returns a list of group elements.
    - get_order(): Returns the order of the group.
    - get_permutation_matrix(h, k): Returns the k x k permutation matrix for element h.
    - get_irreps_info(): Returns a list of dictionaries, each containing:
        - 'name': str (e.g., 'trivial', 'sign', 'E1')
        - 'dim': int (dimension of the irrep)
        - 'character_map': dict (mapping group element to its character value for this irrep)
    """
    def __init__(self):
        self._elements = []
        self._order = 0

    def get_elements(self):
        return self._elements

    def get_order(self):
        return self._order

    def get_permutation_matrix(self, h, k):
        """
        Returns the k x k permutation matrix for group element h.
        This method needs to be implemented by subclasses based on how
        the group acts on a k-dimensional vector space (e.g., a window of k tokens).
        """
        raise NotImplementedError("Subclasses must implement get_permutation_matrix.")

    def get_irreps_info(self):
        """
        Returns a list of dictionaries, each describing an irreducible representation (irrep).
        Each dict should contain 'name', 'dim', and 'character_map'.
        """
        raise NotImplementedError("Subclasses must implement get_irreps_info.")

class CyclicGroup(BaseGroup):
    """
    Represents the Cyclic Group C_n (or Z_n).
    Elements are integers 0 to n-1, representing rotations.
    The action is a cyclic shift.
    """
    def __init__(self, n: int):
        super().__init__()
        if n < 1:
            raise ValueError("n must be a positive integer for CyclicGroup.")
        self.n = n
        self._elements = list(range(n)) # Elements are 0, 1, ..., n-1
        self._order = n

    def get_permutation_matrix(self, h: int, k: int):
        """
        Returns the k x k permutation matrix for a cyclic shift by h.
        Assumes the group acts on k elements, where k is a multiple of n,
        or k=n, and the action is a simple cyclic shift.
        For simplicity, we assume k=n for direct interpretation of cyclic shifts.
        If k != n, the action needs to be clearly defined (e.g., block-wise).
        Here, we assume k=n and h is the number of positions to shift.
        """
        if k != self.n:
            raise ValueError(f"For CyclicGroup(n), k must be equal to n for this permutation matrix definition. Got k={k}, n={self.n}")
        P = np.zeros((k, k), dtype=np.float32)
        for i in range(k):
            P[i, (i - h) % k] = 1 # Shift element at (i-h) to i. Corresponds to x_{h^{-1}(i)}
        return torch.from_numpy(P)

    def get_irreps_info(self):
        """
        Returns irreps info for C_n.
        For real-valued data, complex irreps are often combined into 2D real irreps.
        For C2 (Z2), it's simple: two 1D real irreps.
        For general C_n (n > 2), this would involve more complex logic for 2D real irreps.
        """
        if self.n == 2: # Z2 group
            # Trivial representation (symmetric)
            trivial_char = {0: 1, 1: 1} # h=0 (identity), h=1 (reflection)
            # Sign representation (anti-symmetric)
            sign_char = {0: 1, 1: -1}
            return [
                {'name': 'trivial', 'dim': 1, 'character_map': trivial_char},
                {'name': 'sign', 'dim': 1, 'character_map': sign_char}
            ]
        else:
            # Placeholder for general C_n. This requires handling complex characters
            # and combining conjugate pairs for real representations.
            # For a full implementation, refer to group representation theory texts.
            print(f"Warning: Full real irreps for C_{self.n} are complex and not fully implemented here.")
            print("Returning only trivial irrep as a placeholder.")
            trivial_char = {i: 1 for i in range(self.n)}
            return [
                {'name': 'trivial', 'dim': 1, 'character_map': trivial_char}
            ]

class DihedralGroup(BaseGroup):
    """
    Represents the Dihedral Group D_n, symmetries of a regular n-gon.
    Order is 2n.
    Elements: n rotations (r^0, ..., r^(n-1)) and n reflections (s, sr, ..., sr^(n-1)).
    The action is on k=n vertices of the polygon.
    """
    def __init__(self, n: int):
        super().__init__()
        if n < 2:
            raise ValueError("n must be at least 2 for DihedralGroup.")
        self.n = n
        # Elements represented as tuples (type, value): ('r', rotation_amount) or ('s', reflection_axis_offset)
        self._elements = [('r', i) for i in range(n)] + [('s', i) for i in range(n)]
        self._order = 2 * n

    def get_permutation_matrix(self, h_tuple: tuple, k: int):
        """
        Returns the k x k permutation matrix for a Dihedral group element.
        Assumes k=n (action on vertices of n-gon).
        h_tuple: ('r', i) for rotation by i, or ('s', i) for reflection about axis i.
        """
        if k != self.n:
            raise ValueError(f"For DihedralGroup(n), k must be equal to n for this permutation matrix definition. Got k={k}, n={self.n}")

        P = np.zeros((k, k), dtype=np.float32)
        h_type, h_val = h_tuple

        if h_type == 'r': # Rotation
            for i in range(k):
                P[i, (i - h_val) % k] = 1 # Rotate element at (i-h_val) to i
        elif h_type == 's': # Reflection
            # This is a simplified reflection. For a general D_n, reflections
            # depend on whether n is even/odd and the axis.
            # Here, we assume a reflection across an axis passing through vertex 0 and (n/2) if n is even,
            # or midpoint of edge (0,1) if n is odd, and then rotated.
            # A common way to define reflection for D_n is s_i(j) = (i - j) mod n, or (i + j) mod n
            # Let's use a standard reflection: s(j) = (n - j) % n for j != 0, and s(0)=0.
            # Then apply rotation.
            # For simplicity, we'll use a reflection that flips around a "center"
            # and then apply the rotation offset h_val.
            for i in range(k):
                # Reflection around an axis, then shift by h_val
                # A simple reflection for D_n is mapping i to (n-i) % n.
                # The h_val then indicates which reflection axis.
                # For D_n, there are n reflections.
                # r^j * s: (i) -> (j-i) mod n
                # s * r^j: (i) -> (n - (i+j)) mod n
                reflected_idx = (h_val - i) % k # Simplified reflection: reflects around h_val
                P[i, reflected_idx] = 1
        else:
            raise ValueError(f"Unknown dihedral element type: {h_type}")
        return torch.from_numpy(P)

    def get_irreps_info(self):
        """
        Returns irreps info for D_n.
        For D2 (Klein four-group), it has four 1D real irreps.
        For general D_n (n > 2), this involves 1D and 2D real irreps.
        """
        if self.n == 2: # D2 group (Klein four-group, isomorphic to Z2 x Z2)
            # Elements: ('r',0), ('r',1), ('s',0), ('s',1)
            # ('r',0) = identity (e)
            # ('r',1) = rotation by 180 degrees (r)
            # ('s',0) = reflection (s)
            # ('s',1) = reflection * rotation (sr)
            e, r, s, sr = ('r',0), ('r',1), ('s',0), ('s',1)

            # Four 1D irreps:
            # A1 (trivial): all map to 1
            A1_char = {e: 1, r: 1, s: 1, sr: 1}
            # A2 (sign for rotation, trivial for reflection)
            A2_char = {e: 1, r: 1, s: -1, sr: -1}
            # B1 (trivial for rotation, sign for reflection)
            B1_char = {e: 1, r: -1, s: 1, sr: -1}
            # B2 (sign for both)
            B2_char = {e: 1, r: -1, s: -1, sr: 1}

            return [
                {'name': 'A1', 'dim': 1, 'character_map': A1_char},
                {'name': 'A2', 'dim': 1, 'character_map': A2_char},
                {'name': 'B1', 'dim': 1, 'character_map': B1_char},
                {'name': 'B2', 'dim': 1, 'character_map': B2_char}
            ]
        else:
            # Placeholder for general D_n. This requires handling 2D real irreps.
            # For a full implementation, refer to group representation theory texts.
            print(f"Warning: Full real irreps for D_{self.n} are complex and not fully implemented here.")
            print("Returning only trivial irrep as a placeholder.")
            trivial_char = {h: 1 for h in self._elements}
            return [
                {'name': 'trivial', 'dim': 1, 'character_map': trivial_char}
            ]

# Example Usage (for testing/demonstration)
if __name__ == '__main__':
    print("--- Testing CyclicGroup(2) (Z2) ---")
    z2_group = CyclicGroup(2)
    print(f"Elements: {z2_group.get_elements()}, Order: {z2_group.get_order()}")
    irreps_z2 = z2_group.get_irreps_info()
    for irrep in irreps_z2:
        print(f"Irrep: {irrep['name']}, Dim: {irrep['dim']}, Characters: {irrep['character_map']}")

    # Test permutation matrix for Z2
    k_size = 2 # For Z2, k=n=2 is the natural action
    P_e = z2_group.get_permutation_matrix(0, k_size) # Identity
    P_s = z2_group.get_permutation_matrix(1, k_size) # Reflection
    print(f"Permutation matrix for identity (h=0):\n{P_e}")
    print(f"Permutation matrix for reflection (h=1):\n{P_s}")

    test_vec = torch.tensor([[1.0, 2.0], [3.0, 4.0]]) # Batch=1, k=2, dim=2
    print(f"Original vector:\n{test_vec}")
    print(f"Reflected vector (P_s @ test_vec):\n{P_s @ test_vec}")


    print("\n--- Testing DihedralGroup(2) (D2 - Klein Four-Group) ---")
    d2_group = DihedralGroup(2)
    print(f"Elements: {d2_group.get_elements()}, Order: {d2_group.get_order()}")
    irreps_d2 = d2_group.get_irreps_info()
    for irrep in irreps_d2:
        print(f"Irrep: {irrep['name']}, Dim: {irrep['dim']}, Characters: {irrep['character_map']}")

    # Test permutation matrices for D2
    k_size_d2 = 2 # For D2, k=n=2 is the natural action on vertices
    P_e_d2 = d2_group.get_permutation_matrix(('r', 0), k_size_d2)
    P_r_d2 = d2_group.get_permutation_matrix(('r', 1), k_size_d2)
    P_s_d2 = d2_group.get_permutation_matrix(('s', 0), k_size_d2)
    P_sr_d2 = d2_group.get_permutation_matrix(('s', 1), k_size_d2)

    print(f"Permutation matrix for identity ('r',0):\n{P_e_d2}")
    print(f"Permutation matrix for rotation ('r',1):\n{P_r_d2}")
    print(f"Permutation matrix for reflection ('s',0):\n{P_s_d2}")
    print(f"Permutation matrix for reflection+rotation ('s',1):\n{P_sr_d2}")

    # Test vector for D2
    test_vec_d2 = torch.tensor([[10.0, 20.0], [30.0, 40.0]]) # Batch=1, k=2, dim=2
    print(f"Original vector D2:\n{test_vec_d2}")
    print(f"Rotated vector D2 (P_r_d2 @ test_vec_d2):\n{P_r_d2 @ test_vec_d2}")
    print(f"Reflected vector D2 (P_s_d2 @ test_vec_d2):\n{P_s_d2 @ test_vec_d2}")
