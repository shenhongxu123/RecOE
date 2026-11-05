"""Checkpoint-based reconstruction scores for test-time subspace ablations."""

import torch


SCORE_MODE_DESCRIPTIONS = {
    "dual": "e_id - e_ood (dual-subspace baseline)",
    "id_only": "e_id (ID-subspace reconstruction error only)",
    "ood_only": "-e_ood (negative OOD-subspace reconstruction error only)",
}


def add_score_mode_argument(parser):
    parser.add_argument(
        "--score_mode",
        choices=tuple(SCORE_MODE_DESCRIPTIONS),
        default="dual",
        help=(
            "Test-time scoring ablation on the same checkpoint: "
            "dual = e_id - e_ood; id_only = e_id; ood_only = -e_ood. "
            "All modes use larger scores for OOD and the saved EMA means."
        ),
    )


def reconstruction_error(z_centered: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """Squared residual from the checkpoint's orthonormal subspace."""
    proj = (z_centered @ U) @ U.t()
    diff = z_centered - proj
    return (diff * diff).sum(dim=1)


@torch.no_grad()
def reconstruction_score(
    features: torch.Tensor,
    subspace: torch.nn.Module,
    score_mode: str = "dual",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return OOD-oriented scores without updating checkpoint parameters/buffers.

    Normalize features before centering, as in the existing RecOE tests. Each
    single-subspace mode only evaluates its selected basis and saved EMA mean.
    OOD-only scores are negated because small OOD reconstruction errors indicate
    OOD samples. This sign is fixed in advance, never chosen using test labels.
    """
    if score_mode not in SCORE_MODE_DESCRIPTIONS:
        raise ValueError("Unsupported score_mode: {}".format(score_mode))

    # Normalize before branch-specific centering to retain the training geometry.
    z = features / (features.norm(p=2, dim=1, keepdim=True) + eps)

    if score_mode in ("dual", "id_only"):
        e_id = reconstruction_error(z - subspace.z_id_mean_ema, subspace.U_id)
        if score_mode == "id_only":
            return e_id

    e_ood = reconstruction_error(z - subspace.z_ood_mean_ema, subspace.U_ood)
    if score_mode == "ood_only":
        return -e_ood
    return e_id - e_ood
