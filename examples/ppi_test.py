import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import defaultdict

# Import PSEAD components from the new modular structure
from psead.attention import PSEADAttention
from psead.groups import CyclicGroup, DihedralGroup


# --- PSEAD-Enhanced PPI Classification Model ---


class PSEADTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, group_type, window_size, dropout=0.1):
        super().__init__()
        # Instantiate the correct group based on type
        if group_type == "Z2":
            group = CyclicGroup(window_size)
        elif group_type == "D2":
            group = DihedralGroup(window_size)
        elif group_type == "C4":
            group = CyclicGroup(window_size)  # C4 is a CyclicGroup
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


class PSEADInteractionClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_heads,
        num_layers,
        group_type,
        window_size,
        num_classes=2,  # Binary: Interact/No Interact
        viral_host_classifier_dim=2,  # Binary: Viral-like/Host-like (for interacting pairs)
    ):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.psead_layers = nn.ModuleList(
            [
                PSEADTransformerBlock(hidden_dim, num_heads, group_type, window_size)
                for _ in range(num_layers)
            ]
        )
        # Classifier for Interaction (Interact/No Interact)
        self.interaction_output_layer = nn.Linear(
            hidden_dim * 2, num_classes
        )  # *2 because we concatenate features from two proteins

        # Classifier for Viral-Host vs Host-Host (only for interacting pairs)
        self.viral_host_classifier = nn.Linear(
            hidden_dim * 2, viral_host_classifier_dim
        )

    def forward(self, protein1_obs, protein2_obs):
        # protein_obs: (batch_size, window_size, feature_dim)

        # Embed observations
        embedded_p1 = self.embedding(protein1_obs)
        embedded_p2 = self.embedding(protein2_obs)

        # Store irrep contributions from each PSEADAttention layer
        # Aggregate across layers and proteins for a single average contribution per irrep per batch
        all_irrep_contributions = defaultdict(lambda: {"sum_val": 0.0, "count": 0})

        # Process through PSEAD layers
        for layer in self.psead_layers:
            embedded_p1, irrep_contribs_p1 = layer(embedded_p1)
            embedded_p2, irrep_contribs_p2 = layer(embedded_p2)

            # Aggregate irrep contributions from current layer
            for name, val in irrep_contribs_p1.items():
                all_irrep_contributions[name]["sum_val"] += val
                all_irrep_contributions[name]["count"] += 1
            for name, val in irrep_contribs_p2.items():
                all_irrep_contributions[name]["sum_val"] += val
                all_irrep_contributions[name]["count"] += 1

        # Average irrep contributions across all layers and proteins in the batch
        averaged_irrep_contributions = {
            name: data["sum_val"] / data["count"]
            for name, data in all_irrep_contributions.items()
            if data["count"] > 0
        }

        # Global average pooling for interaction classification
        pooled_p1 = embedded_p1.mean(dim=1)  # (batch_size, hidden_dim)
        pooled_p2 = embedded_p2.mean(dim=1)  # (batch_size, hidden_dim)

        # Concatenate features for interaction prediction
        combined_features = torch.cat((pooled_p1, pooled_p2), dim=-1)

        # Predict interaction status (interact/no interact)
        interaction_logits = self.interaction_output_layer(combined_features)

        # Predict viral-host vs host-host if interaction is predicted
        viral_host_logits = self.viral_host_classifier(combined_features)

        return interaction_logits, viral_host_logits, averaged_irrep_contributions


# --- Simulated Data Generator for PPI ---


class PPIDataGenerator:
    def __init__(self, input_dim, window_size, num_samples_per_class=100):
        self.input_dim = input_dim
        self.window_size = window_size
        self.num_samples = num_samples_per_class

        # Define specific "irrep profiles" for each type of interaction
        # These are designed to correlate with the mock irrep contributions from PSEADAttention
        # (which are fixed based on random.seed(42) in PSEADAttention).
        # You'd need to run PSEADAttention once to see its fixed mock contributions
        # and then adjust these targets accordingly for a strong correlation.
        # For this example, let's assume 'sign' and 'B2' are "viral-like" and 'trivial' and 'A1' are "host-like".
        # These values are arbitrary for demonstration.
        self.irrep_profiles = {
            "host_host": {
                "trivial": 0.6,  # High trivial contribution
                "sign": 0.1,
                "A1": 0.7,  # High A1 contribution for D2 host
                "A2": 0.1,
                "B1": 0.1,
                "B2": 0.1,
                "rot_2fold": 0.1,
                "E_mock1": 0.1,
            },
            "viral_host": {
                "trivial": 0.1,
                "sign": 0.6,  # High sign contribution (Z2)
                "A1": 0.1,
                "A2": 0.1,
                "B1": 0.1,
                "B2": 0.7,  # High B2 contribution (D2)
                "rot_2fold": 0.6,  # High rot_2fold (C4)
                "E_mock1": 0.5,
            },
        }

    def _generate_patch(self, interaction_type="no_interact", group_type=None):
        patch = torch.randn(self.window_size, self.input_dim) * 0.1  # Base noise

        # Embed a pattern that correlates with specific irrep contributions
        # This is a highly simplified way to make the data "symmetry-aware" for the mock.
        # In a real scenario, this would be based on actual protein structural motifs.
        if interaction_type == "host_host":
            # Make features correlate with 'trivial' and 'A1' irreps
            if group_type == "Z2" and self.window_size == 2:
                patch[0] += 0.5
                patch[1] += 0.5  # Symmetric pattern
            elif group_type == "D2" and self.window_size == 2:
                patch[0] += 0.8
                patch[1] += 0.8  # A1-like pattern
        elif interaction_type == "viral_host":
            if group_type == "Z2" and self.window_size == 2:
                patch[0] += 0.5
                patch[1] -= 0.5  # Anti-symmetric pattern
            elif group_type == "D2" and self.window_size == 2:
                patch[0] += 0.8
                patch[1] -= 0.8  # B2-like pattern
            elif group_type == "C4" and self.window_size == 4:
                patch[0] += 0.7
                patch[1] -= 0.7
                patch[2] += 0.7
                patch[
                    3
                ] -= 0.7  # A pattern that might activate 'rot_2fold' or 'E_mock1'

        return patch

    def generate_data(self):
        data = (
            []
        )  # List of (protein1_patch, protein2_patch, interaction_label, specificity_label)

        # 0: No Interaction
        for _ in range(self.num_samples):
            p1 = self._generate_patch(interaction_type="no_interact")
            p2 = self._generate_patch(interaction_type="no_interact")
            data.append((p1, p2, 0, -1))  # -1 for specificity as it's non-interacting

        # 1: Interaction
        # 1.1: Host-Host Interaction (common symmetric motif, e.g., Z2 for both)
        for _ in range(self.num_samples):
            p1 = self._generate_patch(interaction_type="host_host", group_type="Z2")
            p2 = self._generate_patch(interaction_type="host_host", group_type="Z2")
            data.append((p1, p2, 1, 0))  # 0 for host-host

        # 1.2: Viral-Host Interaction (unique symmetric motif, e.g., D2 for viral, C4 for host it binds to)
        for _ in range(self.num_samples):
            p1_viral = self._generate_patch(
                interaction_type="viral_host", group_type="D2"
            )  # Viral protein with D2-like motif
            p2_host = self._generate_patch(
                interaction_type="viral_host", group_type="C4"
            )  # Host protein it binds to with C4-like motif
            data.append((p1_viral, p2_host, 1, 1))  # 1 for viral-host

        random.shuffle(data)
        # Separate into tensors
        p1_list, p2_list, interaction_labels, specificity_labels = zip(*data)
        return (
            torch.stack(p1_list),
            torch.stack(p2_list),
            torch.tensor(interaction_labels),
            torch.tensor(specificity_labels),
        )


# --- Main Execution ---


def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model parameters
    input_feature_dim = 16  # Simplified feature dimension for patches
    hidden_dim = 32
    num_heads = 4
    num_layers = 2
    window_size = (
        4  # Size of the protein patch (e.g., number of residues in an interface)
    )
    # Note: This window_size must match the 'n' of the group used in PSEADAttention.
    # For D2, n=2. For C4, n=4. This will cause issues if not matched.
    # For this test, let's set window_size to 2 for simplicity and use D2 or Z2.
    # If you want to use C4, you must set window_size=4.

    # Let's adjust window_size and group_type to be consistent for the main model.
    # We'll use window_size = 2 and group_type = "D2" for the main classifier.
    # The PPIDataGenerator will still generate different group_type patterns.
    window_size_model = 2
    group_type_model = "D2"  # This is the group PSEADAttention will use internally

    num_interaction_classes = 2  # Interact / No Interact
    num_specificity_classes = 2  # Host-Host / Viral-Host

    # Instantiate the model
    model = PSEADInteractionClassifier(
        input_dim=input_feature_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        group_type=group_type_model,
        window_size=window_size_model,
        num_classes=num_interaction_classes,
        viral_host_classifier_dim=num_specificity_classes,
    ).to(device)

    # Data generation
    num_samples_per_class = 200  # For each of No Interaction, Host-Host, Viral-Host
    # IMPORTANT: The data generator's window_size must match the model's window_size_model
    # and the group types used in data generation must be compatible with PSEADAttention's
    # expectation (i.e., their 'n' must match the window_size).
    # For this setup, we need to ensure the data generator outputs patches of window_size_model.
    data_generator = PPIDataGenerator(
        input_feature_dim, window_size_model, num_samples_per_class
    )
    p1_data, p2_data, interaction_labels, specificity_labels = (
        data_generator.generate_data()
    )

    # Split data (simple 80/20 split)
    total_samples = p1_data.shape[0]
    train_size = int(0.8 * total_samples)

    p1_train, p1_test = p1_data[:train_size].to(device), p1_data[train_size:].to(device)
    p2_train, p2_test = p2_data[:train_size].to(device), p2_data[train_size:].to(device)
    inter_labels_train, inter_labels_test = interaction_labels[:train_size].to(
        device
    ), interaction_labels[train_size:].to(device)
    spec_labels_train, spec_labels_test = specificity_labels[:train_size].to(
        device
    ), specificity_labels[train_size:].to(device)

    # Loss functions and optimizer
    criterion_inter = nn.CrossEntropyLoss()
    criterion_spec = nn.CrossEntropyLoss(
        ignore_index=-1
    )  # Ignore non-interacting pairs for specificity
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Training Loop (Simplified) ---
    print("\n--- Starting Simulated Training ---")
    num_epochs = 20  # Increased epochs for better learning on synthetic data
    batch_size = 32

    for epoch in range(num_epochs):
        model.train()
        epoch_loss_inter = 0
        epoch_loss_spec = 0
        num_batches = 0

        # Create batches
        indices = torch.randperm(train_size)
        for i in range(0, train_size, batch_size):
            batch_indices = indices[i : i + batch_size]

            p1_batch = p1_train[batch_indices]
            p2_batch = p2_train[batch_indices]
            inter_label_batch = inter_labels_train[batch_indices]
            spec_label_batch = spec_labels_train[batch_indices]

            optimizer.zero_grad()

            inter_logits, spec_logits, _ = model(
                p1_batch, p2_batch
            )  # _ to ignore irrep_contributions during training forward

            loss_inter = criterion_inter(inter_logits, inter_label_batch)

            # For specificity loss, only consider interacting pairs (label != -1)
            interacting_mask = inter_label_batch == 1
            if interacting_mask.any():
                loss_spec = criterion_spec(
                    spec_logits[interacting_mask], spec_label_batch[interacting_mask]
                )
            else:
                loss_spec = torch.tensor(0.0).to(
                    device
                )  # No interacting pairs in batch

            total_loss = loss_inter + loss_spec  # Simple weighted sum
            total_loss.backward()
            optimizer.step()

            epoch_loss_inter += loss_inter.item()
            epoch_loss_spec += loss_spec.item()
            num_batches += 1

        avg_loss_inter = epoch_loss_inter / num_batches
        avg_loss_spec = epoch_loss_spec / num_batches
        print(
            f"Epoch {epoch+1}/{num_epochs}, Inter Loss: {avg_loss_inter:.4f}, Spec Loss: {avg_loss_spec:.4f}"
        )

    # --- Evaluation ---
    print("\n--- Starting Simulated Evaluation ---")
    model.eval()
    with torch.no_grad():
        inter_preds_list, spec_preds_list = [], []
        inter_true_list, spec_true_list = [], []

        # Store irrep contributions per specificity class for detailed analysis
        irrep_contribs_by_spec = defaultdict(lambda: defaultdict(float))
        irrep_counts_by_spec = defaultdict(lambda: defaultdict(int))

        for i in range(0, p1_test.shape[0], batch_size):
            p1_batch = p1_test[i : i + batch_size]
            p2_batch = p2_test[i : i + batch_size]
            inter_label_batch = inter_labels_test[i : i + batch_size]
            spec_label_batch = spec_labels_test[i : i + batch_size]

            inter_logits, spec_logits, avg_irrep_contribs_batch = model(
                p1_batch, p2_batch
            )

            inter_preds_list.append(torch.argmax(inter_logits, dim=1).cpu())
            inter_true_list.append(inter_label_batch.cpu())

            # For specificity evaluation, only consider actual interacting pairs
            actual_interacting_mask = inter_label_batch == 1
            if actual_interacting_mask.any():
                spec_preds_list.append(
                    torch.argmax(spec_logits[actual_interacting_mask], dim=1).cpu()
                )
                spec_true_list.append(
                    spec_label_batch[actual_interacting_mask].cpu()
                )  # Only include relevant labels

                # Aggregate irrep contributions for detailed analysis by specificity
                # This loop iterates through each sample in the batch
                for batch_idx in range(p1_batch.shape[0]):
                    if inter_label_batch[batch_idx] == 1:  # Only for interacting pairs
                        current_spec_label = spec_label_batch[batch_idx].item()
                        # Use the batch-averaged irrep contributions for each sample in the batch
                        # This is a simplification for the mock, as true irrep contributions
                        # would be per-sample.
                        for irrep_name, contrib_val in avg_irrep_contribs_batch.items():
                            irrep_contribs_by_spec[current_spec_label][
                                irrep_name
                            ] += contrib_val
                            irrep_counts_by_spec[current_spec_label][irrep_name] += 1

        # Calculate overall metrics
        inter_preds = torch.cat(inter_preds_list)
        inter_true = torch.cat(inter_true_list)
        inter_accuracy = (inter_preds == inter_true).float().mean().item()
        print(f"Interaction Prediction Accuracy: {inter_accuracy:.4f}")

        if len(spec_preds_list) > 0:  # Ensure there were interacting pairs in test set
            spec_preds = torch.cat(spec_preds_list)
            spec_true = torch.cat(spec_true_list)
            spec_accuracy = (spec_preds == spec_true).float().mean().item()
            print(
                f"Specificity (Viral-Host vs Host-Host) Accuracy (on interacting pairs): {spec_accuracy:.4f}"
            )
        else:
            print("No interacting pairs in test set for specificity evaluation.")

        # --- Symmetry Specificity Score Analysis (Detailed) ---
        print("\n--- Averaged Irrep Contributions by Specificity Class ---")
        for spec_label, irrep_data in irrep_contribs_by_spec.items():
            label_name = (
                "Host-Host"
                if spec_label == 0
                else "Viral-Host" if spec_label == 1 else "Unknown"
            )
            print(f"  Class: {label_name}")
            for irrep_name, sum_val in irrep_data.items():
                if irrep_counts_by_spec[spec_label][irrep_name] > 0:
                    avg_contrib = sum_val / irrep_counts_by_spec[spec_label][irrep_name]
                    print(f"    {irrep_name}: {avg_contrib:.4f}")

        # Example of a conceptual "Symmetry Specificity Score"
        # We'll use 'B2' (from D2) as a hypothetical viral-specific irrep
        # and 'A1' (from D2) as a hypothetical host-specific irrep.
        # This is based on the D2 group used by the main model.

        viral_specific_irrep_name = "B2"
        host_specific_irrep_name = "A1"

        avg_viral_host_B2 = irrep_contribs_by_spec[1].get(
            viral_specific_irrep_name, 0.0
        ) / irrep_counts_by_spec[1].get(viral_specific_irrep_name, 1)
        avg_host_host_B2 = irrep_contribs_by_spec[0].get(
            viral_specific_irrep_name, 0.0
        ) / irrep_counts_by_spec[0].get(viral_specific_irrep_name, 1)

        avg_viral_host_A1 = irrep_contribs_by_spec[1].get(
            host_specific_irrep_name, 0.0
        ) / irrep_counts_by_spec[1].get(host_specific_irrep_name, 1)
        avg_host_host_A1 = irrep_contribs_by_spec[0].get(
            host_specific_irrep_name, 0.0
        ) / irrep_counts_by_spec[0].get(host_specific_irrep_name, 1)

        print("\n--- Conceptual Viral-Specific Symmetry Score ---")
        print(
            f"Avg '{viral_specific_irrep_name}' in Viral-Host: {avg_viral_host_B2:.4f}"
        )
        print(f"Avg '{viral_specific_irrep_name}' in Host-Host: {avg_host_host_B2:.4f}")

        print(
            f"Avg '{host_specific_irrep_name}' in Viral-Host: {avg_viral_host_A1:.4f}"
        )
        print(f"Avg '{host_specific_irrep_name}' in Host-Host: {avg_host_host_A1:.4f}")

        # A simple ratio to indicate higher specificity for viral-host
        if avg_host_host_B2 > 1e-6:
            specificity_ratio_B2 = avg_viral_host_B2 / avg_host_host_B2
            print(
                f"Ratio ('{viral_specific_irrep_name}' Viral-Host / Host-Host): {specificity_ratio_B2:.4f}"
            )
            print(
                "Higher ratio suggests this irrep is more prominent in viral-host interactions."
            )
        else:
            print(
                f"Cannot compute specificity ratio for '{viral_specific_irrep_name}' (denominator is zero)."
            )

        if avg_viral_host_A1 > 1e-6:
            specificity_ratio_A1 = avg_host_host_A1 / avg_viral_host_A1
            print(
                f"Ratio ('{host_specific_irrep_name}' Host-Host / Viral-Host): {specificity_ratio_A1:.4f}"
            )
            print(
                "Higher ratio suggests this irrep is more prominent in host-host interactions."
            )
        else:
            print(
                f"Cannot compute specificity ratio for '{host_specific_irrep_name}' (denominator is zero)."
            )

    print("\nSimulated PSEAD application to PPIs completed.")


if __name__ == "__main__":
    import os

    if not os.path.exists("psead"):
        os.makedirs("psead")
    with open("psead/__init__.py", "w") as f:
        f.write("# PSEAD package\n")

    main()
