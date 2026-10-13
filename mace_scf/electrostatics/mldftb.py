"""Non-self-consistent ML-DFTB with a shared s/p electronic basis.

Inputs follow MACE's batched graph dictionary, with graph-level N_alpha,
N_beta and elec_temp. Temperatures default to Kelvin; all energies are eV.
A minimal construction is ``MLDFTB([1, 8], [1., 6.], n_s=1, n_p=1)``;
pass ``model(batch, training=True)`` when fitting energies and forces. Effective
nuclear charges are model parameters chosen with the electron counts, not
necessarily atomic numbers. Use double precision for the electrostatic kernels.
Periodic Hamiltonians are real Gamma-point matrices. Positive elec_temp is
required for force training; zero-width energies and ordinary forces are supported.

The Hamiltonian is evaluated once; the returned electrostatic potential is
diagnostic and is not fed back into the electronic state.

``fragment_diagonalize=True`` replaces the full eigensolve by divide-and-conquer
diagonalisation of overlapping atom-centred fragments of radius ``fragment_radius``
(Å); see ``fragment_state.py``. Saved models can switch with
``model.use_fragment_diagonalization(radius)`` (``radius=None`` restores full).
``batched_diagonalize=True`` keeps full diagonalisation but solves all graphs of a batch in
one padded batched eigh (``batched_state.py``); switch with ``model.use_batched_diagonalization()``.
"""

import numpy as np
import torch
from e3nn import o3
from mace.modules import (
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
)
from .localsources import _LocalSourceModelBase
from .matrix_ops import HamiltonianBuilder, ElectronicState
from .fragment_state import FragmentElectronicState
from .batched_state import BatchedElectronicState
from .density_electrostatics import DensityElectrostatics


def _polar_feature_index(irreps):
    """Columns of the l=0 scalar and l=1 polar-vector channels, grouped by irrep."""
    columns, order = {}, []
    offset = 0
    for mul, ir in o3.Irreps(irreps):
        width = mul * ir.dim
        if ir.l <= 1 and ir.p == (-1) ** ir.l:
            if ir not in columns:
                columns[ir] = []
                order.append(ir)
            columns[ir].extend(range(offset, offset + width))
        offset += width
    selected = o3.Irreps([(len(columns[ir]) // ir.dim, ir) for ir in order])
    index = [column for ir in order for column in columns[ir]]
    return selected, index


def _as_multipole(charges, dipoles):
    """Charge plus dipole in the node-feature frame (q, y, z, x)."""
    if dipoles.ndim != 2 or dipoles.shape[-1] != 3:
        raise ValueError("atomic dipoles must have shape [n_atoms, 3]")
    charges = charges.reshape(dipoles.shape[0], 1)
    return torch.cat((charges, dipoles[:, [1, 2, 0]]), dim=-1)


class MLDFTB(_LocalSourceModelBase):
    def __init__(
        self,
        atomic_numbers,
        effective_nuclear_charges,
        n_s=1,
        n_p=1,
        r_max=5.0,
        hidden_irreps="16x0e + 16x1o",
        num_interactions=2,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=2,
        correlation=2,
        MLP_irreps="16x0e",
        avg_num_neighbors=8.0,
        atomic_energies=None,
        radial_MLP=None,
        radial_type="bessel",
        interaction_cls=RealAgnosticResidualInteractionBlock,
        interaction_cls_first=RealAgnosticInteractionBlock,
        gate=torch.nn.functional.silu,
        heads=None,
        edge_mode="bilinear",
        onsite_mode="quadratic",
        matrix_feature_multiplicity=4,
        matrix_radial_hidden=32,
        elec_temp_units="kelvin",
        energy_kind="internal",
        sigma=1.0,
        pp_scalar="g",
        keep_quadrupoles=False,
        coupling_mode="paired",
        C=None,
        nuclear_profile="shared_g",
        nuclear_charge_mode="per_species",
        self_policy="full",
        pbc_handling="auto",
        kspace_cutoff_factor=1.5,
        hamiltonian_cutoff=None,
        fragment_diagonalize=False,
        fragment_radius=6.0,
        batched_diagonalize=False,
        use_nonlocal_energy=False,
    ):
        super().__init__()
        if fragment_diagonalize and batched_diagonalize:
            raise ValueError("Choose at most one of fragment_diagonalize and batched_diagonalize")
        if energy_kind not in {"internal", "free"}:
            raise ValueError("energy_kind must be internal or free")
        if num_interactions < 1:
            raise ValueError("At least one interaction is required")
        if len(effective_nuclear_charges) != len(atomic_numbers):
            raise ValueError("Supply one effective nuclear charge per species")
        hamiltonian_cutoff = (
            r_max if hamiltonian_cutoff is None else float(hamiltonian_cutoff)
        )
        if not np.isfinite(hamiltonian_cutoff) or not 0 < hamiltonian_cutoff <= r_max:
            raise ValueError(
                "hamiltonian_cutoff must be positive and no larger than r_max"
            )
        heads = heads or ["Default"]
        if atomic_energies is None:
            atomic_energies = np.zeros((len(heads), len(atomic_numbers)))
        self._init_local_model(
            r_max,
            num_bessel,
            num_polynomial_cutoff,
            max_ell,
            interaction_cls,
            interaction_cls_first,
            num_interactions,
            len(atomic_numbers),
            o3.Irreps(hidden_irreps),
            o3.Irreps(MLP_irreps),
            np.asarray(atomic_energies),
            avg_num_neighbors,
            atomic_numbers,
            correlation,
            gate,
            radial_MLP,
            radial_type,
            heads,
        )
        self.hamiltonian = HamiltonianBuilder(
            hidden_irreps,
            n_s,
            n_p,
            edge_mode,
            hamiltonian_cutoff,
            num_bessel,
            matrix_feature_multiplicity,
            matrix_radial_hidden,
            onsite_mode,
        )
        self.electronic_state = ElectronicState(n_s + 3 * n_p, elec_temp_units)
        if fragment_diagonalize:
            self.use_fragment_diagonalization(fragment_radius)
        if batched_diagonalize:
            self.use_batched_diagonalization()
        self.density_electrostatics = DensityElectrostatics(
            n_s,
            n_p,
            effective_nuclear_charges,
            sigma,
            pp_scalar,
            keep_quadrupoles,
            coupling_mode,
            C,
            nuclear_profile,
            nuclear_charge_mode,
            self_policy,
            pbc_handling,
            kspace_cutoff_factor,
        )
        self.energy_kind = energy_kind
        self.use_nonlocal_energy = bool(use_nonlocal_energy)
        self.nonlocal_irreps, feature_index = _polar_feature_index(self.hidden_irreps)
        if self.nonlocal_irreps.dim == 0:
            raise ValueError("nonlocal energy needs l=0 or l=1 node features")
        self.register_buffer(
            "nonlocal_feature_index",
            torch.tensor(feature_index, dtype=torch.long),
            persistent=False,
        )
        multipole_irreps = o3.Irreps("1x0e + 1x1o")
        self.spin_up_multipole_embedding = o3.Linear(
            multipole_irreps, self.nonlocal_irreps
        )
        self.spin_down_multipole_embedding = o3.Linear(
            multipole_irreps, self.nonlocal_irreps
        )
        n_invariants = 2 * sum(mul * mul for mul, _ir in self.nonlocal_irreps)
        self.nonlocal_mlp = torch.nn.Sequential(
            torch.nn.Linear(n_invariants, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 1),
        )

    def use_fragment_diagonalization(self, radius=6.0):
        """Divide-and-conquer diagonalisation with fragment radius in Å; None restores full."""
        state = self.electronic_state
        block, units = state.block_size, state.occupations.elec_temp_units
        tolerance = state.occupations.degeneracy_tolerance
        self.electronic_state = (
            ElectronicState(block, units, tolerance)
            if radius is None
            else FragmentElectronicState(block, radius, units, tolerance)
        )
        return self

    def use_batched_diagonalization(self, batched=True):
        """Full diagonalisation in one batched eigh per batch; False restores the per-graph loop."""
        state = self.electronic_state
        block, units = state.block_size, state.occupations.elec_temp_units
        tolerance = state.occupations.degeneracy_tolerance
        self.electronic_state = (BatchedElectronicState if batched else ElectronicState)(
            block, units, tolerance
        )
        return self
    
    def _multipole_invariants(self, node_features, embedding):
        """d_kk'l = sum_m h_klm p_k'lm for each polar irrep. Returns [n_atoms, sum_l K_l^2]."""
        parts = []
        offset = 0
        for mul, ir in self.nonlocal_irreps:
            dim = ir.dim
            width = mul * dim
            h = node_features[:, offset : offset + width].reshape(-1, mul, dim)
            p = embedding[:, offset : offset + width].reshape(-1, mul, dim)
            parts.append(
                torch.einsum("nkm,nqm->nkq", h, p).reshape(h.shape[0], mul * mul)
            )
            offset += width
        return torch.cat(parts, dim=-1)

    def nonlocal_energy(
        self,
        node_features,
        spin_up_atomic_charges,
        spin_down_atomic_charges,
        spin_up_atomic_dipoles,
        spin_down_atomic_dipoles,
        batch,
    ):
        """Per-graph nonlocal energy, [n_graphs].

        node_features: [n_atoms, n_features]
        spin charges: [n_atoms]
        spin dipoles: [n_atoms, 3] in Cartesian (x, y, z)
        batch: [n_atoms] graph index
        """
        if node_features.shape[-1] != self.hidden_irreps.dim:
            raise ValueError("node_features do not match hidden_irreps")
        if batch.shape != node_features.shape[:1]:
            raise ValueError("batch must contain one graph index per atom")
        node_features = node_features.index_select(-1, self.nonlocal_feature_index)
        embedded_up = self.spin_up_multipole_embedding(
            _as_multipole(spin_up_atomic_charges, spin_up_atomic_dipoles)
        )
        embedded_down = self.spin_down_multipole_embedding(
            _as_multipole(spin_down_atomic_charges, spin_down_atomic_dipoles)
        )
        invariants = torch.cat(
            (
                self._multipole_invariants(node_features, embedded_up),
                self._multipole_invariants(node_features, embedded_down),
            ),
            dim=-1,
        )
        atomic_energy = self.nonlocal_mlp(invariants).squeeze(-1)
        n_graphs = int(batch.max()) + 1
        return atomic_energy.new_zeros(n_graphs).index_add(0, batch, atomic_energy)

    @torch.enable_grad()
    def forward(
        self,
        data,
        training=False,
        compute_force=True,
        compute_virials=False,
        compute_stress=False,
        compute_potential=False,
        return_density=False,
        return_electronic_state=False,
    ):
        if not isinstance(data, dict):
            data = data.to_dict()
        positions = data["positions"]
        if compute_force and not positions.requires_grad:
            positions = positions.clone().requires_grad_(True)
        batch = data.get(
            "batch",
            torch.zeros(len(positions), dtype=torch.long, device=positions.device),
        )
        ptr = data.get(
            "ptr", torch.tensor([0, len(positions)], device=positions.device)
        )
        ng = ptr.numel() - 1
        if bool((ptr[1:] <= ptr[:-1]).any()):
            raise ValueError("Empty graphs are not supported")
        expected_batch = torch.repeat_interleave(
            torch.arange(ng, device=batch.device), ptr[1:] - ptr[:-1]
        )
        if (
            int(ptr[0]) != 0
            or int(ptr[-1]) != len(positions)
            or not torch.equal(batch, expected_batch)
        ):
            raise ValueError("batch and ptr must describe contiguous, nonempty graphs")
        cell = data.get("cell", positions.new_zeros(ng, 3, 3)).reshape(ng, 3, 3)
        pbc = data.get(
            "pbc", torch.zeros(ng, 3, dtype=torch.bool, device=positions.device)
        ).reshape(ng, 3)
        displacement = positions.new_zeros(
            ng, 3, 3, requires_grad=compute_virials or compute_stress
        )
        strain = (displacement + displacement.mT) / 2
        x = positions + torch.einsum("ni,nij->nj", positions, strain[batch])
        cells = cell + cell @ strain
        edges = data["edge_index"]
        if "unit_shifts" in data:
            shifts = torch.einsum(
                "ni,nij->nj", data["unit_shifts"].to(x), cells[batch[edges[0]]]
            )
        else:
            raw = data.get("shifts", x.new_zeros(edges.shape[1], 3))
            shifts = raw + torch.einsum("ni,nij->nj", raw, strain[batch[edges[0]]])
        geometry = dict(positions=x, cell=cells, pbc=pbc, ptr=ptr, batch=batch)
        attrs = data["node_attrs"]
        species = attrs.argmax(-1)

        def graph_field(name, default=None):
            value = data.get(name, default)
            if value is None:
                raise ValueError(f"Missing required graph field {name}")
            value = torch.as_tensor(value, device=x.device, dtype=x.dtype).reshape(-1)
            if value.numel() != ng:
                raise ValueError(f"{name} must contain one value per graph")
            return value

        alpha, beta = graph_field("N_alpha"), graph_field("N_beta")
        temperature = graph_field("elec_temp", x.new_zeros(ng))
        node_heads = data.get(
            "head", torch.zeros(ng, dtype=torch.long, device=x.device)
        ).reshape(-1)[batch]
        node_energy = (
            self.atomic_energies_fn(attrs).gather(1, node_heads[:, None]).squeeze(-1)
        )
        features = self.node_embedding(attrs)
        vectors = x[edges[1]] - x[edges[0]] + shifts
        lengths = vectors.norm(dim=-1, keepdim=True)
        edge_attrs = self.spherical_harmonics(vectors[:, [1, 2, 0]])
        edge_features, _ = self.radial_embedding(
            lengths, attrs, edges, self.atomic_numbers
        )
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            features, sc = interaction(
                node_attrs=attrs,
                node_feats=features,
                edge_attrs=edge_attrs,
                edge_feats=edge_features,
                edge_index=edges,
            )
            features = product(node_feats=features, sc=sc, node_attrs=attrs)
            node_energy = node_energy + readout(features, node_heads).gather(
                1, node_heads[:, None]
            ).squeeze(-1)
        local = x.new_zeros(ng).index_add(0, batch, node_energy)
        H = self.hamiltonian(features, x, edges, shifts, ptr)
        if isinstance(self.electronic_state, FragmentElectronicState):
            state = self.electronic_state(H, alpha, beta, temperature, geometry)
        else:
            state = self.electronic_state(H, alpha, beta, temperature)
        result = self.density_electrostatics(
            state["gamma"],
            geometry,
            species,
            state["gamma_spin"],
            data.get("external_field"),
            data.get("effective_nuclear_charges"),
            compute_potential=compute_potential,
            return_density=return_density,
        )
        expected_charge = (
            x.new_zeros(ng).index_add(0, batch, result["effective_nuclear_charges"])
            - alpha
            - beta
        )
        if "total_charge" in data and not torch.allclose(
            expected_charge, graph_field("total_charge"), atol=1e-5, rtol=1e-5
        ):
            val = graph_field("total_charge")
            raise ValueError(
                f"N_alpha + N_beta is inconsistent with nuclear and total charge, {expected_charge}, {val}"
            )
        energy = local + state["band_energy"] + result["electrostatic_energy"]
        if self.energy_kind == "free":
            energy = energy + state["entropy_energy"]
        energy = energy + x.sum() * 0
        nonlocal_energy = None
        if self.use_nonlocal_energy:
            q_up = self.density_electrostatics.density_map(
                0.5 * (state["gamma"] + state["gamma_spin"])
            )
            q_down = self.density_electrostatics.density_map(
                0.5 * (state["gamma"] - state["gamma_spin"])
            )
            n_up, d_up = self.density_electrostatics.basis.moments(q_up)
            n_down, d_down = self.density_electrostatics.basis.moments(q_down)
            nonlocal_energy = self.nonlocal_energy(
                features, -n_up, -n_down, -d_up, -d_down, batch
            )
            energy = energy + nonlocal_energy

        result.update(
            energy=energy,
            local_energy=local,
            band_energy=state["band_energy"],
            entropy_energy=state["entropy_energy"],
            chemical_potentials=state["chemical_potentials"],
            count_residuals=state["count_residuals"],
            forces=None,
            virials=None,
            stress=None,
        )
        if nonlocal_energy is not None:
            result["nonlocal_energy"] = nonlocal_energy
        targets = ([positions] if compute_force else []) + (
            [displacement] if compute_virials or compute_stress else []
        )
        if targets:
            grads = torch.autograd.grad(
                energy.sum(), targets, create_graph=training, retain_graph=True
            )
            if compute_force:
                result["forces"] = -grads[0]
            if compute_virials or compute_stress:
                result["virials"] = -grads[-1]
            if compute_stress:
                volume = torch.linalg.det(cell).abs()
                result["stress"] = torch.where(
                    (volume > 1e-10)[:, None, None],
                    grads[-1] / volume.clamp_min(1e-10)[:, None, None],
                    torch.zeros_like(grads[-1]),
                )
        if return_electronic_state:
            result.update(hamiltonians=H, **state)
        return result


NonSCFModel = MLDFTB
