# DualSubspaceClass.py
# Keep the original Riemannian/Stiefel objective, but progressively increase the
# optimized subspace rank in an Oja++-style schedule: h/2, h/4, h/8, ...

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt

from typing import Dict, Optional, Tuple
_VALID_EXPANSION_MODES = {"half_remaining", "quarter_total", "one_by_one"}
# Main progressive module used by the CIFAR and ImageNet training entry points.
# Additional experimental classes below remain intact for source compatibility.
class DualSubspaceModule(nn.Module):


    def __init__(
        self,
        h_dim: int,
        z_dim: int,
        eps: float = 1e-6,
        ema_momentum: float = 0.99,
        init_rank: Optional[int] = None,
        expansion_mode: str = "half_remaining",
        h_dim_id: Optional[int] = None,
        h_dim_ood: Optional[int] = None,
        init_rank_id: Optional[int] = None,
        init_rank_ood: Optional[int] = None,
    ):
        super().__init__()
        h_dim_id = h_dim if h_dim_id is None else h_dim_id
        h_dim_ood = h_dim if h_dim_ood is None else h_dim_ood
        for name, value in (("h_dim_id", h_dim_id), ("h_dim_ood", h_dim_ood)):
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
            if z_dim < value:
                raise ValueError(
                    f"{name}={value} cannot exceed z_dim={z_dim}; "
                    f"Stiefel(z_dim, {name}) requires z_dim >= {name}."
                )

        self.h_dim_id = int(h_dim_id)
        self.h_dim_ood = int(h_dim_ood)
        # Kept for loading code written for equal-dimensional checkpoints.
        self.h_dim = max(self.h_dim_id, self.h_dim_ood)
        self.z_dim = int(z_dim)
        self.eps = float(eps)
        self.ema_momentum = float(ema_momentum)
        self.expansion_mode = str(expansion_mode)

        if init_rank_id is None:
            init_rank_id = init_rank
        if init_rank_ood is None:
            init_rank_ood = init_rank
        if init_rank_id is None:
            init_rank_id = max(1, math.ceil(self.h_dim_id / 2))
        if init_rank_ood is None:
            init_rank_ood = max(1, math.ceil(self.h_dim_ood / 2))
        self.active_rank_id = int(max(1, min(init_rank_id, self.h_dim_id)))
        self.active_rank_ood = int(max(1, min(init_rank_ood, self.h_dim_ood)))
        self.active_rank = max(self.active_rank_id, self.active_rank_ood)

        self.register_buffer("z_id_mean_ema", torch.zeros(1, z_dim, requires_grad=False))
        self.register_buffer("z_ood_mean_ema", torch.zeros(1, z_dim, requires_grad=False))
        self.register_buffer("ema_initialized", torch.tensor(False, dtype=torch.bool))

        self.U_id = geoopt.ManifoldParameter(
            self._orthogonal_random(self.z_dim, self.active_rank_id),
            manifold=geoopt.manifolds.Stiefel(),
        )
        self.U_ood = geoopt.ManifoldParameter(
            self._orthogonal_random(self.z_dim, self.active_rank_ood),
            manifold=geoopt.manifolds.Stiefel(),
        )

    @staticmethod
    def _orthogonal_random(rows: int, cols: int, device=None, dtype=None) -> torch.Tensor:
        X = torch.empty(rows, cols, device=device, dtype=dtype)
        nn.init.normal_(X)
        Q, _ = torch.linalg.qr(X, mode="reduced")
        return Q[:, :cols].contiguous()

    @staticmethod
    def _orthogonal_append(U_old: torch.Tensor, n_new: int) -> torch.Tensor:
        """
        Append n_new random columns orthogonal to U_old and return a full
        orthonormal matrix [U_old, U_new].
        """
        if n_new <= 0:
            return U_old.detach().clone()

        z_dim, old_rank = U_old.shape
        G = torch.randn(z_dim, n_new, device=U_old.device, dtype=U_old.dtype)

        if old_rank > 0:
            G = G - U_old @ (U_old.t() @ G)

        Q_new, R = torch.linalg.qr(G, mode="reduced")
        if bool((torch.diagonal(R).abs() < 1e-8).any()):
            G = torch.randn(z_dim, n_new, device=U_old.device, dtype=U_old.dtype)
            if old_rank > 0:
                G = G - U_old @ (U_old.t() @ G)
            Q_new, _ = torch.linalg.qr(G, mode="reduced")

        U_cat = torch.cat([U_old.detach(), Q_new[:, :n_new]], dim=1)
        U_cat, _ = torch.linalg.qr(U_cat, mode="reduced")
        return U_cat.contiguous()

    def _next_rank(self, active_rank: int, h_dim: int) -> int:
        if active_rank >= h_dim:
            return h_dim
        remaining = h_dim - active_rank
        if self.expansion_mode == "one_by_one":
            add = 1
        elif self.expansion_mode == "quarter_total":
            add = max(1, math.ceil(h_dim / 4))
        else:
            # Oja++-style: add half of the remaining dimensions.
            # Example h=80: 40 -> 60 -> 70 -> 75 -> 78 -> 79 -> 80.
            add = max(1, math.ceil(remaining / 2))

        return min(h_dim, active_rank + add)

    def next_ranks(self):
        return (
            self._next_rank(self.active_rank_id, self.h_dim_id),
            self._next_rank(self.active_rank_ood, self.h_dim_ood),
        )

    def next_rank(self) -> int:
        """Backward-compatible scalar view of the next ranks."""
        return max(self.next_ranks())

    @torch.no_grad()
    def expand_rank(self, new_rank: Optional[int] = None) -> bool:
        """
        Increase active rank while preserving learned columns.

        Returns True if expansion happened. After this call, the caller must
        rebuild the Riemannian optimizer, because U_id/U_ood Parameter objects
        have been replaced with larger ManifoldParameters.
        """
        return self.expand_ranks(new_rank_id=new_rank, new_rank_ood=new_rank)

    @torch.no_grad()
    def expand_ranks(
        self,
        new_rank_id: Optional[int] = None,
        new_rank_ood: Optional[int] = None,
    ) -> bool:
        """Expand U_id and U_ood independently while preserving learned columns."""
        next_id, next_ood = self.next_ranks()
        if new_rank_id is None:
            new_rank_id = next_id
        if new_rank_ood is None:
            new_rank_ood = next_ood
        new_rank_id = int(max(self.active_rank_id, min(new_rank_id, self.h_dim_id)))
        new_rank_ood = int(max(self.active_rank_ood, min(new_rank_ood, self.h_dim_ood)))

        expanded = False
        if new_rank_id > self.active_rank_id:
            U_id_new = self._orthogonal_append(self.U_id.data, new_rank_id - self.active_rank_id)
            self.U_id = geoopt.ManifoldParameter(U_id_new, manifold=geoopt.manifolds.Stiefel())
            self.active_rank_id = new_rank_id
            expanded = True
        if new_rank_ood > self.active_rank_ood:
            U_ood_new = self._orthogonal_append(self.U_ood.data, new_rank_ood - self.active_rank_ood)
            self.U_ood = geoopt.ManifoldParameter(U_ood_new, manifold=geoopt.manifolds.Stiefel())
            self.active_rank_ood = new_rank_ood
            expanded = True

        # active_rank is the legacy scalar view; branch ranks may differ.
        self.active_rank = max(self.active_rank_id, self.active_rank_ood)
        return expanded

    def _reconstruction_error(self, z: torch.Tensor, U: torch.Tensor, squared: bool = True):
        zU = z @ U
        proj = zU @ U.t()
        diff = z - proj
        if squared:
            return (diff * diff).sum(dim=1)
        return diff.norm(p=2, dim=1)

    def orthogonality_penalty(self):
        M = self.U_id.t() @ self.U_ood
        return (M * M).sum()

    def margin_losses(
        self,
        z_id: torch.Tensor,
        z_ood: torch.Tensor,
        margin: float = 1.0,
        squared: bool = True,
    ):
        z_id = z_id / (z_id.norm(p=2, dim=1, keepdim=True) + self.eps)
        z_ood = z_ood / (z_ood.norm(p=2, dim=1, keepdim=True) + self.eps)

        current_z_id_mean = z_id.mean(dim=0, keepdim=True)
        current_z_ood_mean = z_ood.mean(dim=0, keepdim=True)

        if self.training:
            with torch.no_grad():
                if not bool(self.ema_initialized.item()):
                    self.z_id_mean_ema.copy_(current_z_id_mean.detach())
                    self.z_ood_mean_ema.copy_(current_z_ood_mean.detach())
                    self.ema_initialized.fill_(True)
                else:
                    self.z_id_mean_ema.mul_(self.ema_momentum).add_(
                        current_z_id_mean.detach(), alpha=1.0 - self.ema_momentum
                    )
                    self.z_ood_mean_ema.mul_(self.ema_momentum).add_(
                        current_z_ood_mean.detach(), alpha=1.0 - self.ema_momentum
                    )

        # Fit and rejection terms use current batch centers, not the EMA buffers.
        z_id_mean = current_z_id_mean
        z_ood_mean = current_z_ood_mean

        z_id_id = z_id - z_id_mean
        z_id_ood = z_id - z_ood_mean
        z_ood_ood = z_ood - z_ood_mean
        z_ood_id = z_ood - z_id_mean

        e_id_on_id = self._reconstruction_error(z_id_id, self.U_id, squared=squared)
        e_id_on_ood = self._reconstruction_error(z_id_ood, self.U_ood, squared=squared)
        e_ood_on_ood = self._reconstruction_error(z_ood_ood, self.U_ood, squared=squared)
        e_ood_on_id = self._reconstruction_error(z_ood_id, self.U_id, squared=squared)

        loss_fit_id = e_id_on_id.mean()
        loss_fit_ood = e_ood_on_ood.mean()

        loss_sep_id = F.relu(margin - e_ood_on_id).mean()
        loss_sep_ood = F.relu(margin - e_id_on_ood).mean()

        loss_sep = loss_fit_id + loss_fit_ood + loss_sep_id + loss_sep_ood

        stats = {
            "active_rank": torch.as_tensor(self.active_rank, device=z_id.device),
            "active_rank_id": torch.as_tensor(self.active_rank_id, device=z_id.device),
            "active_rank_ood": torch.as_tensor(self.active_rank_ood, device=z_id.device),
            "e_id_on_id": e_id_on_id.detach().mean(),
            "e_ood_on_id": e_ood_on_id.detach().mean(),
            "e_ood_on_ood": e_ood_on_ood.detach().mean(),
            "e_id_on_ood": e_id_on_ood.detach().mean(),
            "loss_fit_id": loss_fit_id.detach(),
            "loss_fit_ood": loss_fit_ood.detach(),
            "loss_sep_id": loss_sep_id.detach(),
            "loss_sep_ood": loss_sep_ood.detach(),
        }
        return loss_sep, stats

    def forward(
        self,
        z_id: torch.Tensor,
        z_ood: torch.Tensor,
        beta_ortho: float = 0.1,
        margin: float = 1.0,
        squared: bool = True,
    ):
        loss_sep, stats = self.margin_losses(z_id, z_ood, margin=margin, squared=squared)
        loss_ortho = self.orthogonality_penalty()
        loss_total = loss_sep + beta_ortho * loss_ortho
        stats.update({
            "loss_sep": loss_sep.detach(),
            "loss_ortho": loss_ortho.detach(),
            "loss_total": loss_total.detach(),
        })
        return loss_total, stats


class SeparateDimsDualSubspaceModule(DualSubspaceModule):
    """Oja++ dual subspaces with independent ID and OOD rank schedules.

    Unlike :class:`DualSubspaceModule`, this interface has no shared
    ``h_dim`` or ``init_rank`` argument.  Callers must configure the two
    subspaces explicitly, which prevents an equal-rank fallback from being
    applied accidentally.

    Each basis is independently constrained to ``Stiefel(z_dim, rank)``.
    Full cross-orthogonality is possible only when
    ``h_dim_id + h_dim_ood <= z_dim``; larger totals are supported, but the
    cross-subspace penalty then has a non-zero theoretical minimum.
    """

    def __init__(
        self,
        h_dim_id: int,
        h_dim_ood: int,
        z_dim: int,
        eps: float = 1e-6,
        ema_momentum: float = 0.99,
        init_rank_id: Optional[int] = None,
        init_rank_ood: Optional[int] = None,
        expansion_mode: str = "half_remaining",
    ):
        super().__init__(
            h_dim=max(int(h_dim_id), int(h_dim_ood)),
            z_dim=z_dim,
            eps=eps,
            ema_momentum=ema_momentum,
            expansion_mode=expansion_mode,
            h_dim_id=h_dim_id,
            h_dim_ood=h_dim_ood,
            init_rank_id=init_rank_id,
            init_rank_ood=init_rank_ood,
        )

    @property
    def target_ranks(self):
        return self.h_dim_id, self.h_dim_ood

    @property
    def active_ranks(self):
        return self.active_rank_id, self.active_rank_ood

    def ranks_full(self) -> bool:
        return (
            self.active_rank_id >= self.h_dim_id
            and self.active_rank_ood >= self.h_dim_ood
        )

    def _rank_normalized_reconstruction_error(
        self,
        z: torch.Tensor,
        U: torch.Tensor,
        squared: bool = True,
    ):
        """Normalize residual energy by its isotropic rank expectation.

        For a random isotropic vector and an orthonormal rank-r basis, the
        expected squared residual fraction is ``1 - r / z_dim``.  Dividing by
        this factor makes reconstruction errors from differently sized
        subspaces comparable.
        """
        zU = z @ U
        proj = zU @ U.t()
        diff = z - proj

        if squared:
            error = diff.square().sum(dim=1)
        else:
            error = diff.norm(p=2, dim=1)

        rank = int(U.shape[1])
        residual_fraction = max(1.0 - float(rank) / float(self.z_dim), self.eps)
        rank_scale = residual_fraction if squared else math.sqrt(residual_fraction)
        return error / rank_scale

    def margin_losses(
        self,
        z_id: torch.Tensor,
        z_ood: torch.Tensor,
        margin: float = 1.0,
        margin_id: Optional[float] = None,
        margin_ood: Optional[float] = None,
        squared: bool = True,
    ):
        """Compute fit and rejection losses with two independent margins.

        ``margin_id`` constrains OOD features measured against ``U_id``;
        ``margin_ood`` constrains ID features measured against ``U_ood``.
        The shared ``margin`` remains as a backward-compatible fallback.
        """
        margin_id = float(margin if margin_id is None else margin_id)
        margin_ood = float(margin if margin_ood is None else margin_ood)

        z_id = z_id / (z_id.norm(p=2, dim=1, keepdim=True) + self.eps)
        z_ood = z_ood / (z_ood.norm(p=2, dim=1, keepdim=True) + self.eps)

        current_z_id_mean = z_id.mean(dim=0, keepdim=True)
        current_z_ood_mean = z_ood.mean(dim=0, keepdim=True)

        if self.training:
            with torch.no_grad():
                if not bool(self.ema_initialized.item()):
                    self.z_id_mean_ema.copy_(current_z_id_mean.detach())
                    self.z_ood_mean_ema.copy_(current_z_ood_mean.detach())
                    self.ema_initialized.fill_(True)
                else:
                    self.z_id_mean_ema.mul_(self.ema_momentum).add_(
                        current_z_id_mean.detach(), alpha=1.0 - self.ema_momentum
                    )
                    self.z_ood_mean_ema.mul_(self.ema_momentum).add_(
                        current_z_ood_mean.detach(), alpha=1.0 - self.ema_momentum
                    )

        z_id_id = z_id - current_z_id_mean
        z_id_ood = z_id - current_z_ood_mean
        z_ood_ood = z_ood - current_z_ood_mean
        z_ood_id = z_ood - current_z_id_mean

        e_id_on_id = self._rank_normalized_reconstruction_error(
            z_id_id, self.U_id, squared=squared
        )
        e_id_on_ood = self._rank_normalized_reconstruction_error(
            z_id_ood, self.U_ood, squared=squared
        )
        e_ood_on_ood = self._rank_normalized_reconstruction_error(
            z_ood_ood, self.U_ood, squared=squared
        )
        e_ood_on_id = self._rank_normalized_reconstruction_error(
            z_ood_id, self.U_id, squared=squared
        )

        loss_fit_id = e_id_on_id.mean()
        loss_fit_ood = e_ood_on_ood.mean()
        loss_sep_id = F.relu(margin_id - e_ood_on_id).mean()
        loss_sep_ood = F.relu(margin_ood - e_id_on_ood).mean()
        loss_sep = loss_fit_id + loss_fit_ood + loss_sep_id + loss_sep_ood

        stats = {
            "active_rank": torch.as_tensor(self.active_rank, device=z_id.device),
            "active_rank_id": torch.as_tensor(self.active_rank_id, device=z_id.device),
            "active_rank_ood": torch.as_tensor(self.active_rank_ood, device=z_id.device),
            "margin_id": torch.as_tensor(margin_id, device=z_id.device),
            "margin_ood": torch.as_tensor(margin_ood, device=z_id.device),
            "e_id_on_id": e_id_on_id.detach().mean(),
            "e_ood_on_id": e_ood_on_id.detach().mean(),
            "e_ood_on_ood": e_ood_on_ood.detach().mean(),
            "e_id_on_ood": e_id_on_ood.detach().mean(),
            "loss_fit_id": loss_fit_id.detach(),
            "loss_fit_ood": loss_fit_ood.detach(),
            "loss_sep_id": loss_sep_id.detach(),
            "loss_sep_ood": loss_sep_ood.detach(),
        }
        return loss_sep, stats

    def forward(
        self,
        z_id: torch.Tensor,
        z_ood: torch.Tensor,
        beta_ortho: float = 0.1,
        margin: float = 1.0,
        margin_id: Optional[float] = None,
        margin_ood: Optional[float] = None,
        squared: bool = True,
    ):
        loss_sep, stats = self.margin_losses(
            z_id,
            z_ood,
            margin=margin,
            margin_id=margin_id,
            margin_ood=margin_ood,
            squared=squared,
        )
        loss_ortho = self.orthogonality_penalty()
        loss_total = loss_sep + beta_ortho * loss_ortho
        stats.update({
            "loss_sep": loss_sep.detach(),
            "loss_ortho": loss_ortho.detach(),
            "loss_total": loss_total.detach(),
        })
        return loss_total, stats



class DualSubspaceModule_ablation(nn.Module):


    def __init__(
        self,
        h_dim: int,
        z_dim: int,
        mode : str,
        eps: float = 1e-6,
        ema_momentum: float = 0.99,
        init_rank: Optional[int] = None,
        expansion_mode: str = "half_remaining",
        h_dim_id: Optional[int] = None,
        h_dim_ood: Optional[int] = None,
        init_rank_id: Optional[int] = None,
        init_rank_ood: Optional[int] = None,
    ):
        super().__init__()
        h_dim_id = h_dim if h_dim_id is None else h_dim_id
        h_dim_ood = h_dim if h_dim_ood is None else h_dim_ood
        for name, value in (("h_dim_id", h_dim_id), ("h_dim_ood", h_dim_ood)):
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
            if z_dim < value:
                raise ValueError(
                    f"{name}={value} cannot exceed z_dim={z_dim}; "
                    f"Stiefel(z_dim, {name}) requires z_dim >= {name}."
                )

        self.h_dim_id = int(h_dim_id)
        self.h_dim_ood = int(h_dim_ood)
        # Kept for loading code written for equal-dimensional checkpoints.
        self.h_dim = max(self.h_dim_id, self.h_dim_ood)
        self.z_dim = int(z_dim)
        self.eps = float(eps)
        self.ema_momentum = float(ema_momentum)
        self.expansion_mode = str(expansion_mode)

        if init_rank_id is None:
            init_rank_id = init_rank
        if init_rank_ood is None:
            init_rank_ood = init_rank
        if init_rank_id is None:
            init_rank_id = max(1, math.ceil(self.h_dim_id / 2))
        if init_rank_ood is None:
            init_rank_ood = max(1, math.ceil(self.h_dim_ood / 2))
        self.active_rank_id = int(max(1, min(init_rank_id, self.h_dim_id)))
        self.active_rank_ood = int(max(1, min(init_rank_ood, self.h_dim_ood)))
        self.active_rank = max(self.active_rank_id, self.active_rank_ood)

        self.register_buffer("z_id_mean_ema", torch.zeros(1, z_dim, requires_grad=False))
        self.register_buffer("z_ood_mean_ema", torch.zeros(1, z_dim, requires_grad=False))
        self.register_buffer("ema_initialized", torch.tensor(False, dtype=torch.bool))

        self.U_id = geoopt.ManifoldParameter(
            self._orthogonal_random(self.z_dim, self.active_rank_id),
            manifold=geoopt.manifolds.Stiefel(),
        )
        self.U_ood = geoopt.ManifoldParameter(
            self._orthogonal_random(self.z_dim, self.active_rank_ood),
            manifold=geoopt.manifolds.Stiefel(),
        )

    @staticmethod
    def _orthogonal_random(rows: int, cols: int, device=None, dtype=None) -> torch.Tensor:
        X = torch.empty(rows, cols, device=device, dtype=dtype)
        nn.init.normal_(X)
        Q, _ = torch.linalg.qr(X, mode="reduced")
        return Q[:, :cols].contiguous()

    @staticmethod
    def _orthogonal_append(U_old: torch.Tensor, n_new: int) -> torch.Tensor:
        """
        Append n_new random columns orthogonal to U_old and return a full
        orthonormal matrix [U_old, U_new].
        """
        if n_new <= 0:
            return U_old.detach().clone()

        z_dim, old_rank = U_old.shape
        G = torch.randn(z_dim, n_new, device=U_old.device, dtype=U_old.dtype)

        if old_rank > 0:
            G = G - U_old @ (U_old.t() @ G)

        Q_new, R = torch.linalg.qr(G, mode="reduced")
        if bool((torch.diagonal(R).abs() < 1e-8).any()):
            G = torch.randn(z_dim, n_new, device=U_old.device, dtype=U_old.dtype)
            if old_rank > 0:
                G = G - U_old @ (U_old.t() @ G)
            Q_new, _ = torch.linalg.qr(G, mode="reduced")

        U_cat = torch.cat([U_old.detach(), Q_new[:, :n_new]], dim=1)
        U_cat, _ = torch.linalg.qr(U_cat, mode="reduced")
        return U_cat.contiguous()

    def _next_rank(self, active_rank: int, h_dim: int) -> int:
        if active_rank >= h_dim:
            return h_dim
        remaining = h_dim - active_rank
        if self.expansion_mode == "one_by_one":
            add = 1
        elif self.expansion_mode == "quarter_total":
            add = max(1, math.ceil(h_dim / 4))
        else:
            # Oja++-style: add half of the remaining dimensions.
            # Example h=80: 40 -> 60 -> 70 -> 75 -> 78 -> 79 -> 80.
            add = max(1, math.ceil(remaining / 2))

        return min(h_dim, active_rank + add)

    def next_ranks(self):
        return (
            self._next_rank(self.active_rank_id, self.h_dim_id),
            self._next_rank(self.active_rank_ood, self.h_dim_ood),
        )

    def next_rank(self) -> int:
        """Backward-compatible scalar view of the next ranks."""
        return max(self.next_ranks())

    @torch.no_grad()
    def expand_rank(self, new_rank: Optional[int] = None) -> bool:
        """
        Increase active rank while preserving learned columns.

        Returns True if expansion happened. After this call, the caller must
        rebuild the Riemannian optimizer, because U_id/U_ood Parameter objects
        have been replaced with larger ManifoldParameters.
        """
        return self.expand_ranks(new_rank_id=new_rank, new_rank_ood=new_rank)

    @torch.no_grad()
    def expand_ranks(
        self,
        new_rank_id: Optional[int] = None,
        new_rank_ood: Optional[int] = None,
    ) -> bool:
        """Expand U_id and U_ood independently while preserving learned columns."""
        next_id, next_ood = self.next_ranks()
        if new_rank_id is None:
            new_rank_id = next_id
        if new_rank_ood is None:
            new_rank_ood = next_ood
        new_rank_id = int(max(self.active_rank_id, min(new_rank_id, self.h_dim_id)))
        new_rank_ood = int(max(self.active_rank_ood, min(new_rank_ood, self.h_dim_ood)))

        expanded = False
        if new_rank_id > self.active_rank_id:
            U_id_new = self._orthogonal_append(self.U_id.data, new_rank_id - self.active_rank_id)
            self.U_id = geoopt.ManifoldParameter(U_id_new, manifold=geoopt.manifolds.Stiefel())
            self.active_rank_id = new_rank_id
            expanded = True
        if new_rank_ood > self.active_rank_ood:
            U_ood_new = self._orthogonal_append(self.U_ood.data, new_rank_ood - self.active_rank_ood)
            self.U_ood = geoopt.ManifoldParameter(U_ood_new, manifold=geoopt.manifolds.Stiefel())
            self.active_rank_ood = new_rank_ood
            expanded = True

        self.active_rank = max(self.active_rank_id, self.active_rank_ood)
        return expanded

    def _reconstruction_error(self, z: torch.Tensor, U: torch.Tensor, squared: bool = True):
        zU = z @ U
        proj = zU @ U.t()
        diff = z - proj
        if squared:
            return (diff * diff).sum(dim=1)
        return diff.norm(p=2, dim=1)

    def orthogonality_penalty(self):
        M = self.U_id.t() @ self.U_ood
        return (M * M).sum()

    def margin_losses(
        self,
        z_id: torch.Tensor,
        z_ood: torch.Tensor,
        margin: float = 1.0,
        squared: bool = True,
        mode: str = "full"
    ):
        z_id = z_id / (z_id.norm(p=2, dim=1, keepdim=True) + self.eps)
        z_ood = z_ood / (z_ood.norm(p=2, dim=1, keepdim=True) + self.eps)

        current_z_id_mean = z_id.mean(dim=0, keepdim=True)
        current_z_ood_mean = z_ood.mean(dim=0, keepdim=True)

        if self.training:
            with torch.no_grad():
                if not bool(self.ema_initialized.item()):
                    self.z_id_mean_ema.copy_(current_z_id_mean.detach())
                    self.z_ood_mean_ema.copy_(current_z_ood_mean.detach())
                    self.ema_initialized.fill_(True)
                else:
                    self.z_id_mean_ema.mul_(self.ema_momentum).add_(
                        current_z_id_mean.detach(), alpha=1.0 - self.ema_momentum
                    )
                    self.z_ood_mean_ema.mul_(self.ema_momentum).add_(
                        current_z_ood_mean.detach(), alpha=1.0 - self.ema_momentum
                    )

        z_id_mean = current_z_id_mean
        z_ood_mean = current_z_ood_mean

        z_id_id = z_id - z_id_mean
        z_id_ood = z_id - z_ood_mean
        z_ood_ood = z_ood - z_ood_mean
        z_ood_id = z_ood - z_id_mean

        e_id_on_id = self._reconstruction_error(z_id_id, self.U_id, squared=squared)
        e_id_on_ood = self._reconstruction_error(z_id_ood, self.U_ood, squared=squared)
        e_ood_on_ood = self._reconstruction_error(z_ood_ood, self.U_ood, squared=squared)
        e_ood_on_id = self._reconstruction_error(z_ood_id, self.U_id, squared=squared)

        loss_fit_id = e_id_on_id.mean()
        loss_fit_ood = e_ood_on_ood.mean()

        loss_sep_id = F.relu(margin - e_ood_on_id).mean()
        loss_sep_ood = F.relu(margin - e_id_on_ood).mean()

        if mode =="full" or mode =="fit" or mode =="wo_ortho":
            loss_sep = loss_fit_id + loss_fit_ood + loss_sep_id + loss_sep_ood
        elif mode =="wo_fit":
            loss_sep = loss_sep_id + loss_sep_ood
        elif mode =="wo_sep":
            loss_sep = loss_fit_id + loss_fit_ood

        stats = {
            "active_rank": torch.as_tensor(self.active_rank, device=z_id.device),
            "active_rank_id": torch.as_tensor(self.active_rank_id, device=z_id.device),
            "active_rank_ood": torch.as_tensor(self.active_rank_ood, device=z_id.device),
            "e_id_on_id": e_id_on_id.detach().mean(),
            "e_ood_on_id": e_ood_on_id.detach().mean(),
            "e_ood_on_ood": e_ood_on_ood.detach().mean(),
            "e_id_on_ood": e_id_on_ood.detach().mean(),
            "loss_fit_id": loss_fit_id.detach(),
            "loss_fit_ood": loss_fit_ood.detach(),
            "loss_sep_id": loss_sep_id.detach(),
            "loss_sep_ood": loss_sep_ood.detach(),
        }
        return loss_sep, stats

    def forward(
        self,
        z_id: torch.Tensor,
        z_ood: torch.Tensor,
        beta_ortho: float = 0.1,
        margin: float = 1.0,
        mode: str = "full",
        squared: bool = True,

    ):
        loss_sep, stats = self.margin_losses(z_id, z_ood, margin=margin, squared=squared, mode=mode)
        loss_ortho = self.orthogonality_penalty()
        if mode =="wo_ortho":
            loss_total = loss_sep
        else:   
            loss_total = loss_sep + beta_ortho * loss_ortho
        stats.update({
            "loss_sep": loss_sep.detach(),
            "loss_ortho": loss_ortho.detach(),
            "loss_total": loss_total.detach(),
        })
        return loss_total, stats



class _SingleSubspaceModule(nn.Module):
    """Base implementation shared by ID-only and OOD-only ablations."""

    subspace_mode: str
    basis_name: str

    def __init__(
        self,
        h_dim: int,
        z_dim: int,
        eps: float = 1e-6,
        ema_momentum: float = 0.99,
        init_rank: Optional[int] = None,
        expansion_mode: str = "half_remaining",
    ) -> None:
        super().__init__()

        if h_dim < 1:
            raise ValueError(f"h_dim must be >= 1, got {h_dim}")
        if z_dim < h_dim:
            raise ValueError(
                f"h_dim={h_dim} cannot exceed z_dim={z_dim}; "
                "Stiefel(z_dim, h_dim) requires z_dim >= h_dim."
            )
        if not 0.0 <= ema_momentum < 1.0:
            raise ValueError(
                f"ema_momentum must be in [0, 1), got {ema_momentum}"
            )
        if expansion_mode not in _VALID_EXPANSION_MODES:
            raise ValueError(
                f"Unsupported expansion_mode={expansion_mode!r}; "
                f"choose from {sorted(_VALID_EXPANSION_MODES)}"
            )

        self.h_dim = int(h_dim)
        self.z_dim = int(z_dim)
        self.eps = float(eps)
        self.ema_momentum = float(ema_momentum)
        self.expansion_mode = str(expansion_mode)

        if init_rank is None:
            init_rank = max(1, math.ceil(self.h_dim / 2))
        self.active_rank = int(max(1, min(int(init_rank), self.h_dim)))

        self.register_buffer(
            "z_id_mean_ema",
            torch.zeros(1, self.z_dim, requires_grad=False),
        )
        self.register_buffer(
            "z_ood_mean_ema",
            torch.zeros(1, self.z_dim, requires_grad=False),
        )
        self.register_buffer(
            "ema_initialized",
            torch.tensor(False, dtype=torch.bool),
        )

        basis = geoopt.ManifoldParameter(
            self._orthogonal_random(self.z_dim, self.active_rank),
            manifold=geoopt.manifolds.Stiefel(),
        )
        setattr(self, self.basis_name, basis)
        self._sync_compatibility_attributes()

    @property
    def basis(self) -> torch.Tensor:
        return getattr(self, self.basis_name)

    @property
    def is_id_subspace(self) -> bool:
        return self.subspace_mode == "single_id"

    def _sync_compatibility_attributes(self) -> None:
        """Expose legacy dual-subspace metadata without creating a fake basis."""
        if self.is_id_subspace:
            self.h_dim_id = self.h_dim
            self.h_dim_ood = 0
            self.active_rank_id = self.active_rank
            self.active_rank_ood = 0
        else:
            self.h_dim_id = 0
            self.h_dim_ood = self.h_dim
            self.active_rank_id = 0
            self.active_rank_ood = self.active_rank

    @staticmethod
    def _orthogonal_random(
        rows: int,
        cols: int,
        device=None,
        dtype=None,
    ) -> torch.Tensor:
        x = torch.empty(rows, cols, device=device, dtype=dtype)
        nn.init.normal_(x)
        q, _ = torch.linalg.qr(x, mode="reduced")
        return q[:, :cols].contiguous()

    @staticmethod
    def _orthogonal_append(u_old: torch.Tensor, n_new: int) -> torch.Tensor:
        """Append columns orthogonal to the existing learned basis."""
        if n_new <= 0:
            return u_old.detach().clone()

        z_dim, old_rank = u_old.shape
        if old_rank + n_new > z_dim:
            raise ValueError(
                f"Cannot expand rank from {old_rank} by {n_new} when z_dim={z_dim}."
            )

        # A full QR on [U_old, random] is more robust than a single projection
        # when several new columns are appended at once.
        random_block = torch.randn(
            z_dim,
            n_new,
            device=u_old.device,
            dtype=u_old.dtype,
        )
        candidate = torch.cat([u_old.detach(), random_block], dim=1)
        q, _ = torch.linalg.qr(candidate, mode="reduced")
        return q[:, : old_rank + n_new].contiguous()

    def _next_rank_value(self) -> int:
        if self.active_rank >= self.h_dim:
            return self.h_dim

        remaining = self.h_dim - self.active_rank
        if self.expansion_mode == "one_by_one":
            add = 1
        elif self.expansion_mode == "quarter_total":
            add = max(1, math.ceil(self.h_dim / 4))
        else:
            add = max(1, math.ceil(remaining / 2))

        return min(self.h_dim, self.active_rank + add)

    def next_rank(self) -> int:
        return self._next_rank_value()

    def next_ranks(self) -> Tuple[int, int]:
        """Legacy-compatible pair where the absent subspace has rank zero."""
        next_rank = self.next_rank()
        return (next_rank, 0) if self.is_id_subspace else (0, next_rank)

    def ranks_full(self) -> bool:
        return self.active_rank >= self.h_dim

    @torch.no_grad()
    def expand_rank(self, new_rank: Optional[int] = None) -> bool:
        if new_rank is None:
            new_rank = self.next_rank()
        new_rank = int(max(self.active_rank, min(int(new_rank), self.h_dim)))

        if new_rank <= self.active_rank:
            return False

        expanded_basis = self._orthogonal_append(
            self.basis.data,
            new_rank - self.active_rank,
        )
        setattr(
            self,
            self.basis_name,
            geoopt.ManifoldParameter(
                expanded_basis,
                manifold=geoopt.manifolds.Stiefel(),
            ),
        )
        self.active_rank = new_rank
        self._sync_compatibility_attributes()
        return True

    @torch.no_grad()
    def expand_ranks(
        self,
        new_rank_id: Optional[int] = None,
        new_rank_ood: Optional[int] = None,
    ) -> bool:
        """Compatibility wrapper used by older progressive training scripts."""
        selected_rank = new_rank_id if self.is_id_subspace else new_rank_ood
        return self.expand_rank(selected_rank)

    def _normalize(self, z: torch.Tensor) -> torch.Tensor:
        return z / (z.norm(p=2, dim=1, keepdim=True) + self.eps)

    def _reconstruction_error(
        self,
        z_centered: torch.Tensor,
        squared: bool = True,
    ) -> torch.Tensor:
        projection = (z_centered @ self.basis) @ self.basis.t()
        residual = z_centered - projection
        if squared:
            return residual.square().sum(dim=1)
        return residual.norm(p=2, dim=1)

    def _batch_means_and_update_ema(
        self,
        z_id: torch.Tensor,
        z_ood: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_id = self._normalize(z_id)
        z_ood = self._normalize(z_ood)

        current_id_mean = z_id.mean(dim=0, keepdim=True)
        current_ood_mean = z_ood.mean(dim=0, keepdim=True)

        if self.training:
            with torch.no_grad():
                if not bool(self.ema_initialized.item()):
                    self.z_id_mean_ema.copy_(current_id_mean.detach())
                    self.z_ood_mean_ema.copy_(current_ood_mean.detach())
                    self.ema_initialized.fill_(True)
                else:
                    momentum = self.ema_momentum
                    self.z_id_mean_ema.mul_(momentum).add_(
                        current_id_mean.detach(),
                        alpha=1.0 - momentum,
                    )
                    self.z_ood_mean_ema.mul_(momentum).add_(
                        current_ood_mean.detach(),
                        alpha=1.0 - momentum,
                    )

        return z_id, z_ood, current_id_mean, current_ood_mean

    def margin_losses(
        self,
        z_id: torch.Tensor,
        z_ood: torch.Tensor,
        margin: float = 1.0,
        squared: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if z_id.ndim != 2 or z_ood.ndim != 2:
            raise ValueError(
                "z_id and z_ood must be rank-2 feature tensors with shape [batch, z_dim]."
            )
        if z_id.shape[1] != self.z_dim or z_ood.shape[1] != self.z_dim:
            raise ValueError(
                f"Expected feature dimension {self.z_dim}, got "
                f"{z_id.shape[1]} and {z_ood.shape[1]}."
            )

        z_id, z_ood, id_mean, ood_mean = self._batch_means_and_update_ema(
            z_id,
            z_ood,
        )

        zero = z_id.new_zeros(())
        stats: Dict[str, torch.Tensor] = {
            "active_rank": torch.as_tensor(self.active_rank, device=z_id.device),
            "active_rank_id": torch.as_tensor(self.active_rank_id, device=z_id.device),
            "active_rank_ood": torch.as_tensor(self.active_rank_ood, device=z_id.device),
            "e_id_on_id": zero.detach(),
            "e_ood_on_id": zero.detach(),
            "e_ood_on_ood": zero.detach(),
            "e_id_on_ood": zero.detach(),
            "loss_fit_id": zero.detach(),
            "loss_fit_ood": zero.detach(),
            "loss_sep_id": zero.detach(),
            "loss_sep_ood": zero.detach(),
        }

        if self.is_id_subspace:
            e_fit = self._reconstruction_error(z_id - id_mean, squared=squared)
            e_opposite = self._reconstruction_error(z_ood - id_mean, squared=squared)
            loss_fit = e_fit.mean()
            loss_reject = F.relu(float(margin) - e_opposite).mean()

            stats.update(
                {
                    "e_id_on_id": e_fit.detach().mean(),
                    "e_ood_on_id": e_opposite.detach().mean(),
                    "loss_fit_id": loss_fit.detach(),
                    "loss_sep_id": loss_reject.detach(),
                }
            )
        else:
            e_fit = self._reconstruction_error(z_ood - ood_mean, squared=squared)
            e_opposite = self._reconstruction_error(z_id - ood_mean, squared=squared)
            loss_fit = e_fit.mean()
            loss_reject = F.relu(float(margin) - e_opposite).mean()

            stats.update(
                {
                    "e_ood_on_ood": e_fit.detach().mean(),
                    "e_id_on_ood": e_opposite.detach().mean(),
                    "loss_fit_ood": loss_fit.detach(),
                    "loss_sep_ood": loss_reject.detach(),
                }
            )

        loss_subspace = loss_fit + loss_reject
        stats.update(
            {
                "loss_fit": loss_fit.detach(),
                "loss_reject": loss_reject.detach(),
            }
        )
        return loss_subspace, stats

    def forward(
        self,
        z_id: torch.Tensor,
        z_ood: torch.Tensor,
        beta_ortho: float = 0.0,
        margin: float = 1.0,
        squared: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # beta_ortho is intentionally ignored: a single-subspace ablation has
        # no cross-subspace orthogonality relation to regularize.
        del beta_ortho

        loss_subspace, stats = self.margin_losses(
            z_id,
            z_ood,
            margin=margin,
            squared=squared,
        )
        loss_ortho = loss_subspace.new_zeros(())
        stats.update(
            {
                "loss_sep": loss_subspace.detach(),
                "loss_ortho": loss_ortho.detach(),
                "loss_total": loss_subspace.detach(),
            }
        )
        return loss_subspace, stats


class SingleIDSubspaceModule(_SingleSubspaceModule):
    """ID-only subspace: fit ID features and reject exposed OOD features."""

    subspace_mode = "single_id"
    basis_name = "U_id"


class SingleOODSubspaceModule(_SingleSubspaceModule):
    """OOD-only subspace: fit exposed OOD features and reject ID features."""

    subspace_mode = "single_ood"
    basis_name = "U_ood"


# Backward-compatible aliases for older experiment scripts.
DualSubspaceModule_ID = SingleIDSubspaceModule
DualSubspaceModule_OOD = SingleOODSubspaceModule


__all__ = [
    "SingleIDSubspaceModule",
    "SingleOODSubspaceModule",
    "DualSubspaceModule_ID",
    "DualSubspaceModule_OOD",
]