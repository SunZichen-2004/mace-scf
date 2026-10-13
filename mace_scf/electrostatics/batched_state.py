"""Full diagonalisation of all graphs of a batch in one padded batched eigh.

Every dense H is padded to the largest graph of the batch with decoupled orbitals far above the
spectrum, so the real eigenpairs of graph g are its lowest n_g*B levels. Per graph and spin a
chemical potential fixes the electron count, and
    gamma_a        = sum_spin U_a diag(f) U_a^T                   on-site density block of atom a
    band energy    = sum_spin sum_k f_k eps_k                     (= Tr P H)
    entropy energy = tau sum_spin sum_k [f ln f + (1-f) ln(1-f)]
The same results as ElectronicState, without its per-graph loop and matrix-function double
backward: derivatives flow through torch.linalg.eigh and a fixed-count Newton correction of mu.
Positive elec_temp is required; exactly degenerate levels make eigh derivatives ill-conditioned.
"""

import torch
from torch import nn

from .fragment_state import PAD_ENERGY, PAD_SPACING, batched_eigh, fit_chemical_potentials
from .matrix_ops import FermiDiracOccupations


def padded_hamiltonians(hamiltonians):
    """Stacked Hamiltonians [ng, M, M] with decoupled padding levels, and real-level mask [ng, M]."""
    sizes = torch.tensor(list(map(len, hamiltonians)), device=hamiltonians[0].device)
    M = int(sizes.max())
    pad_to = lambda H: nn.functional.pad(H, (0, M - len(H), 0, M - len(H)))
    Hp = torch.stack(list(map(pad_to, hamiltonians)))
    real = torch.arange(M, device=Hp.device) < sizes[:, None]
    levels = PAD_ENERGY + PAD_SPACING * torch.arange(M, dtype=Hp.dtype, device=Hp.device)
    return Hp + torch.diag_embed((~real).to(Hp.dtype) * levels), real


class BatchedElectronicState(nn.Module):
    """Drop-in replacement for ElectronicState solving all graphs in one batched eigh."""

    def __init__(self, block_size, elec_temp_units="kelvin", degeneracy_tolerance=1e-9):
        super().__init__()
        self.block_size = block_size
        self.occupations = FermiDiracOccupations(elec_temp_units, degeneracy_tolerance)

    def extra_repr(self):
        return f"block_size={self.block_size}"

    def forward(self, hamiltonians, N_alpha, N_beta, elec_temp):
        B, ng = self.block_size, len(hamiltonians)
        tau = self.occupations.thermal_energy(elec_temp)
        if bool((tau <= 0).any()):
            raise ValueError("Batched diagonalisation requires positive elec_temp")
        counts = torch.stack((N_alpha, N_beta), dim=-1)
        sizes = list(map(len, hamiltonians))

        Hp, real = padded_hamiltonians(hamiltonians)
        eps, U = batched_eigh(Hp)
        del Hp
        w = real.to(eps.dtype)
        mu = fit_chemical_potentials(eps, w, torch.arange(ng, device=eps.device), tau, counts)
        x = (mu[:, None, :] - eps[:, :, None]) / tau[:, None, None]
        f = torch.sigmoid(x) * w[:, :, None]
        f_total, f_spin = f.sum(-1), f[..., 0] - f[..., 1]

        M = eps.shape[1]
        Ua = U.reshape(ng, M // B, B, M)
        atoms = real.reshape(ng, M // B, B)[..., 0]
        onsite = lambda occ: ((Ua * occ[:, None, None, :]) @ Ua.transpose(-1, -2))[atoms]
        entropy_terms = f * nn.functional.logsigmoid(x) + (w[:, :, None] - f) * nn.functional.logsigmoid(-x)
        return dict(
            gamma=onsite(f_total),
            gamma_spin=onsite(f_spin),
            band_energy=(eps * f_total).sum(-1),
            entropy_energy=tau * entropy_terms.sum((1, 2)),
            chemical_potentials=mu,
            count_residuals=f.sum(1) - counts.detach(),
            eigenvalues=list(eps[real].split(sizes)),
            occupations=list(f[real].split(sizes)),
        )
