"""Divide-and-conquer electronic state from overlapping atom-centred fragments of H.

Fragment i is the principal submatrix of H on the atoms within ``radius`` of atom i (minimum
image, centre first), padded to the largest fragment of the batch with decoupled orbitals far
above the spectrum. All fragments of all graphs are solved in one batched eigh. Per graph and
spin a shared chemical potential fixes the sum of central-atom populations,
    sum_ik w_ik f((eps_ik - mu) / tau) = N_spin,   w_ik = sum_{a in centre i} U_ak^2,
and every atom keeps only its own contributions:
    gamma_i        = sum_spin U_c diag(f) U_c^T                    on-site density block
    band energy    = sum_spin sum_ik w_ik f_ik eps_ik              (= sum_{a in i, b} P_ab H_ba)
    entropy energy = tau sum_spin sum_ik w_ik [f ln f + (1-f) ln(1-f)]
The full density matrix is never assembled. Derivatives flow through torch.linalg.eigh and a
fixed-count Newton correction of mu; positive elec_temp is required.
"""

import warnings
from functools import reduce

import torch
from torch import nn

from .matrix_ops import FermiDiracOccupations

PAD_ENERGY = 1.0e3  # eV
PAD_SPACING = 1.0e-2  # eV; nondegenerate padding levels keep eigh derivatives finite
SCAN_POINTS = 64
SCAN_LEVELS = 6
NEWTON_STEPS = 2
# Batched eigh (torch 2.14 / CUDA 13) silently returns wrong eigenpairs for every matrix whose
# offset in the batch exceeds 2**31 elements, so batches are capped at that size.
MAX_EIGH_BATCH_ELEMENTS = 2**31 - 1


def _frame(cell, pbc):
    eye = torch.eye(3, dtype=cell.dtype, device=cell.device)
    return torch.where(pbc[:, None], cell, eye)


def minimum_image_distances(positions, cell, pbc):
    """All-pair minimum-image distances [n, n], exact below half the smallest periodic width."""
    frame = _frame(cell, pbc)
    frac = (positions[:, None, :] - positions[None, :, :]) @ torch.linalg.inv(frame)
    frac = frac - torch.round(frac) * pbc.to(frac.dtype)
    return torch.linalg.norm(frac @ frame, dim=-1)


def periodic_widths(cell, pbc):
    frame = _frame(cell, pbc)
    widths = torch.linalg.det(frame).abs() / torch.linalg.cross(frame[[1, 2, 0]], frame[[2, 0, 1]]).norm(dim=-1)
    return widths[pbc]


def fragment_neighbours(positions, cell, pbc, radius):
    """Atoms within radius of each atom, centre first, padded with -1: [n, M]."""
    widths = periodic_widths(cell, pbc)
    if widths.numel() and 2 * radius >= float(widths.min()):
        warnings.warn(f"fragment radius {radius} Å is not below half the periodic cell width "
                      f"{float(widths.min()):.2f} Å; fragments see only one image of each atom")
    dist = minimum_image_distances(positions, cell, pbc)
    M = int((dist <= radius).sum(1).max())
    order = torch.sort(dist, dim=1, stable=True).indices[:, :M]
    return torch.where(torch.gather(dist, 1, order) <= radius, order, -1)


def graph_fragments(H, nbr, M, B):
    """Fragment Hamiltonians [n, M*B, M*B] (padding still zero) and real-orbital mask [n, M*B]."""
    n = nbr.shape[0]
    nbr = torch.nn.functional.pad(nbr, (0, M - nbr.shape[1]), value=-1)
    mask = nbr >= 0
    atom = torch.where(mask, nbr, n)
    blocks = H.new_zeros(n + 1, n + 1, B, B)
    blocks[:n, :n] = H.reshape(n, B, n, B).transpose(1, 2)
    Hf = blocks[atom[:, :, None], atom[:, None, :]].transpose(2, 3).reshape(n, M * B, M * B)
    return Hf, mask.repeat_interleave(B, dim=1)


def batched_eigh(Hf):
    per_batch = max(1, MAX_EIGH_BATCH_ELEMENTS // (Hf.shape[1] * Hf.shape[2]))
    return tuple(map(torch.cat, zip(*map(torch.linalg.eigh, Hf.split(per_batch)))))


def _counts(mu, eps, w, graph, tau, ng):
    """Central-atom population per graph and spin for mu [ng, 2, ...] -> [ng, 2, ...]."""
    extra = (None,) * (mu.dim() - 2)
    x = (mu[graph][:, None] - eps[(slice(None), slice(None), None) + extra]) / tau[graph][(slice(None), None, None) + extra]
    per_fragment = (w[(slice(None), slice(None), None) + extra] * torch.sigmoid(x)).sum(1)
    return per_fragment.new_zeros((ng,) + per_fragment.shape[1:]).index_add(0, graph, per_fragment)


def _scan(bracket, eps, w, graph, tau, counts):
    lo, hi = bracket
    s = torch.linspace(0, 1, SCAN_POINTS, dtype=eps.dtype, device=eps.device)
    grid = lo[..., None] + (hi - lo)[..., None] * s
    k = (_counts(grid, eps, w, graph, tau, len(counts)) < counts[..., None]).sum(-1, keepdim=True)
    k = k.clamp(1, SCAN_POINTS - 1)
    return grid.gather(-1, k - 1)[..., 0], grid.gather(-1, k)[..., 0]


def _newton(mu, eps, w, graph, tau, counts):
    f = torch.sigmoid((mu[graph][:, None, :] - eps[:, :, None]) / tau[graph][:, None, None])
    ng = len(counts)
    count = f.new_zeros(ng, 2).index_add(0, graph, (w[:, :, None] * f).sum(1))
    slope = f.new_zeros(ng, 2).index_add(0, graph, (w[:, :, None] * f * (1 - f)).sum(1)) / tau[:, None]
    step = (counts - count) / slope.clamp_min(torch.finfo(slope.dtype).tiny)
    # Inside a gap much wider than tau the slope underflows and rounding noise in the count
    # would throw mu out of the spectrum; a step of at most tau keeps it in the gap.
    return mu + torch.maximum(torch.minimum(step, tau[:, None]), -tau[:, None])


def fit_chemical_potentials(eps, w, graph, tau, counts):
    """Shared mu per graph and spin [ng, 2]: nested vectorised scans, then differentiable Newton steps."""
    with torch.no_grad():
        e, ww = eps.detach(), w.detach()
        real = ww > 1e-12
        margin = 50 * float(tau.max()) + 1.0
        lo = torch.full_like(counts, float(e.masked_fill(~real, float("inf")).min()) - margin)
        hi = torch.full_like(counts, float(e.masked_fill(~real, -float("inf")).max()) + margin)
        lo, hi = reduce(lambda b, _: _scan(b, e, ww, graph, tau, counts), range(SCAN_LEVELS), (lo, hi))
        c = _counts(torch.stack((lo, hi), -1), e, ww, graph, tau, len(counts))
        mu = lo + (counts - c[..., 0]) * (hi - lo) / (c[..., 1] - c[..., 0]).clamp_min(1e-300)
        mu = torch.minimum(torch.maximum(mu, lo), hi)
    return reduce(lambda m, _: _newton(m, eps, w, graph, tau, counts), range(NEWTON_STEPS), mu)


class FragmentElectronicState(nn.Module):
    """Drop-in replacement for ElectronicState using divide-and-conquer diagonalisation.

    ``radius`` (Å) is the fragment buffer radius. Fragment eigenvalues [n_g, M*B],
    occupations [n_g, M*B, 2] and central weights replace the global spectrum in the output.
    """

    def __init__(self, block_size, radius=6.0, elec_temp_units="kelvin", degeneracy_tolerance=1e-9):
        super().__init__()
        if not radius > 0:
            raise ValueError("Fragment radius must be positive")
        self.block_size = block_size
        self.radius = float(radius)
        self.occupations = FermiDiracOccupations(elec_temp_units, degeneracy_tolerance)

    def extra_repr(self):
        return f"block_size={self.block_size}, radius={self.radius}"

    def forward(self, hamiltonians, N_alpha, N_beta, elec_temp, geometry):
        B, ng = self.block_size, len(hamiltonians)
        ptr, graph = geometry["ptr"], geometry["batch"]
        positions, cell, pbc = geometry["positions"].detach(), geometry["cell"].detach(), geometry["pbc"]
        tau = self.occupations.thermal_energy(elec_temp)
        if bool((tau <= 0).any()):
            raise ValueError("Fragment diagonalisation requires positive elec_temp")
        counts = torch.stack((N_alpha, N_beta), dim=-1)
        sizes = (ptr[1:] - ptr[:-1]).tolist()

        with torch.no_grad():
            nbrs = list(map(lambda g: fragment_neighbours(positions[ptr[g]:ptr[g + 1]], cell[g], pbc[g], self.radius),
                            range(ng)))
        M = max(map(lambda nb: nb.shape[1], nbrs))
        Hf, omask = map(torch.cat, zip(*map(lambda H, nb: graph_fragments(H, nb, M, B), hamiltonians, nbrs)))
        pad = PAD_ENERGY + PAD_SPACING * torch.arange(M * B, dtype=Hf.dtype, device=Hf.device)
        Hf.diagonal(dim1=1, dim2=2).add_((~omask).to(Hf.dtype) * pad)
        eps, U = batched_eigh(Hf)
        del Hf
        Uc = U[:, :B, :]
        w = (Uc**2).sum(1)

        mu = fit_chemical_potentials(eps, w, graph, tau, counts)
        x = (mu[graph][:, None, :] - eps[:, :, None]) / tau[graph][:, None, None]
        f = torch.sigmoid(x)
        f_total, f_spin = f.sum(-1), f[..., 0] - f[..., 1]
        gamma = (Uc * f_total[:, None, :]) @ Uc.transpose(1, 2)
        gamma_spin = (Uc * f_spin[:, None, :]) @ Uc.transpose(1, 2)
        per_graph = lambda v: v.new_zeros(ng).index_add(0, graph, v)
        band = per_graph((w * eps * f_total).sum(-1))
        entropy_terms = f * torch.nn.functional.logsigmoid(x) + (1 - f) * torch.nn.functional.logsigmoid(-x)
        entropy_energy = tau * per_graph((w[:, :, None] * entropy_terms).sum((1, 2)))
        count = f.new_zeros(ng, 2).index_add(0, graph, (w[:, :, None] * f).sum(1))
        return dict(
            gamma=gamma,
            gamma_spin=gamma_spin,
            band_energy=band,
            entropy_energy=entropy_energy,
            chemical_potentials=mu,
            count_residuals=count - counts.detach(),
            eigenvalues=list(eps.split(sizes)),
            occupations=list(f.split(sizes)),
            fragment_weights=list(w.split(sizes)),
            fragment_neighbours=nbrs,
        )
