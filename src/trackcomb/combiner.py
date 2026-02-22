"""High-level combination engine for event tracks and PV hypotheses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .models import (
    CombinationCuts,
    CombinationResult,
    PrimaryVertex,
    TrackPreselection,
    TrackState,
    iter_n_body_combinations,
)
from .physics import (
    fit_vertex_xyz_t,
    min_impact_parameter_to_pvs,
    pair_kinematics,
    pairwise_doca,
    pairwise_time_chi2,
    sum_lorentz,
    track_to_lorentz,
)


@dataclass
class ParticleCombiner:
    """Build and filter n-body particle candidates from track inputs."""

    def preselect_tracks(
        self,
        tracks: Sequence[TrackState],
        primary_vertices: Sequence[PrimaryVertex],
        preselection: TrackPreselection | None = None,
    ) -> list[TrackState]:
        """Apply track-level preselection before combinatorics."""
        if preselection is None:
            return list(tracks)
        out: list[TrackState] = []
        for t in tracks:
            if preselection.min_pt is not None and t.pt < preselection.min_pt:
                continue
            if preselection.min_eta is not None and t.eta < preselection.min_eta:
                continue
            if preselection.max_eta is not None and t.eta > preselection.max_eta:
                continue
            if preselection.min_ip_to_any_pv is not None:
                min_ip, _, _ = min_impact_parameter_to_pvs(t, list(primary_vertices))
                if min_ip < preselection.min_ip_to_any_pv:
                    continue
            out.append(t)
        return out

    def combine(
        self,
        tracks: Sequence[TrackState],
        primary_vertices: Sequence[PrimaryVertex],
        n_body: int,
        mass_hypotheses: Sequence[Sequence[float]],
        preselection: TrackPreselection | None = None,
        cuts: CombinationCuts | None = None,
    ) -> list[CombinationResult]:
        """Build n-body candidates and apply candidate-level cuts.

        Workflow:
        1. Preselect tracks.
        2. Enumerate n-body combinations.
        3. Fit vertex `(x,y,z,time)` and compute observables.
        4. Apply cuts (doca, chi2, mass, pt, eta, charge pattern).
        5. Return accepted `CombinationResult` objects.
        """
        selected_tracks = self.preselect_tracks(tracks, primary_vertices, preselection)
        if not selected_tracks:
            return []
        pvs = list(primary_vertices)
        if not pvs:
            raise ValueError("At least one primary vertex is required.")
        valid_hypotheses = self._validate_hypotheses(mass_hypotheses, n_body)
        cuts = cuts or CombinationCuts()
        if cuts.allowed_charge_patterns is not None:
            for pat in cuts.allowed_charge_patterns:
                if len(pat) != n_body:
                    raise ValueError(
                        f"Charge pattern '{pat}' length does not match n_body={n_body}."
                    )
                if any(ch not in "+-0" for ch in pat):
                    raise ValueError(
                        f"Charge pattern '{pat}' contains invalid symbol. Use +, -, or 0."
                    )

        results: list[CombinationResult] = []
        for combo in iter_n_body_combinations(selected_tracks, n_body):
            combo_tracks = list(combo)
            vertex_xyz, vertex_time, vertex_chi2, vertex_time_chi2 = fit_vertex_xyz_t(combo_tracks)
            pair_time_chi2 = pairwise_time_chi2(combo_tracks)
            if cuts.max_vertex_chi2 is not None and vertex_chi2 > cuts.max_vertex_chi2:
                continue
            if cuts.max_vertex_time_chi2 is not None and vertex_time_chi2 > cuts.max_vertex_time_chi2:
                continue
            if cuts.max_pair_time_chi2 is not None and pair_time_chi2 > cuts.max_pair_time_chi2:
                continue

            doca_pairs = pairwise_doca(combo_tracks)
            if cuts.max_doca is not None and any(v > cuts.max_doca for v in doca_pairs.values()):
                continue

            vertices_xy = tuple(t.extrapolate(vertex_xyz[2]) for t in combo_tracks)
            track_min_ip: dict[str, float] = {}
            track_min_ip_chi2: dict[str, float] = {}
            track_charges: dict[str, int] = {}
            track_pid_info: dict[str, dict[str, float | bool]] = {}
            best_pv_counts: dict[str, int] = {}
            for track in combo_tracks:
                ip, ip_chi2, pv_id = min_impact_parameter_to_pvs(track, pvs)
                track_min_ip[track.track_id] = ip
                track_min_ip_chi2[track.track_id] = ip_chi2
                track_charges[track.track_id] = int(track.charge)
                track_pid_info[track.track_id] = {
                    "hasRICH1": track.has_rich1,
                    "hasRICH2": track.has_rich2,
                    "richDLL_pi": track.rich_dll_pi,
                    "richDLL_k": track.rich_dll_k,
                    "richDLL_p": track.rich_dll_p,
                    "richDLL_e": track.rich_dll_e,
                    "hasCALO": track.has_calo,
                    "caloDLL_e": track.calo_dll_e,
                }
                if pv_id is not None:
                    best_pv_counts[pv_id] = best_pv_counts.get(pv_id, 0) + 1
            best_pv_id = max(best_pv_counts, key=best_pv_counts.get) if best_pv_counts else None
            charge_pattern = "".join("+" if t.charge > 0 else "-" if t.charge < 0 else "0" for t in combo_tracks)
            total_charge = sum(int(t.charge) for t in combo_tracks)
            if cuts.allowed_charge_patterns is not None and charge_pattern not in cuts.allowed_charge_patterns:
                continue

            for masses in valid_hypotheses:
                p4 = sum_lorentz(
                    track_to_lorentz(track, mass)
                    for track, mass in zip(combo_tracks, masses, strict=True)
                )
                pair_pt, pair_eta = pair_kinematics(p4)
                mass = p4.mass
                if cuts.min_mass is not None and mass < cuts.min_mass:
                    continue
                if cuts.max_mass is not None and mass > cuts.max_mass:
                    continue
                if cuts.min_pair_pt is not None and pair_pt < cuts.min_pair_pt:
                    continue
                if cuts.max_pair_pt is not None and pair_pt > cuts.max_pair_pt:
                    continue
                if cuts.min_pair_eta is not None and pair_eta < cuts.min_pair_eta:
                    continue
                if cuts.max_pair_eta is not None and pair_eta > cuts.max_pair_eta:
                    continue

                results.append(
                    CombinationResult(
                        track_ids=tuple(t.track_id for t in combo_tracks),
                        masses=tuple(masses),
                        vertex_xyz=vertex_xyz,
                        vertex_time=vertex_time,
                        vertices_xy=vertices_xy,
                        candidate_p4=p4,
                        vertex_chi2=vertex_chi2,
                        vertex_time_chi2=vertex_time_chi2,
                        pair_time_chi2=pair_time_chi2,
                        doca_pairs=doca_pairs,
                        track_min_ip=track_min_ip,
                        track_min_ip_chi2=track_min_ip_chi2,
                        track_charges=track_charges,
                        track_pid_info=track_pid_info,
                        charge_pattern=charge_pattern,
                        total_charge=total_charge,
                        pair_pt=pair_pt,
                        pair_eta=pair_eta,
                        best_pv_id=best_pv_id,
                    )
                )
        return results

    @staticmethod
    def _validate_hypotheses(
        mass_hypotheses: Sequence[Sequence[float]], n_body: int
    ) -> list[tuple[float, ...]]:
        """Validate mass-hypothesis shape and coerce to tuples of float."""
        if not mass_hypotheses:
            raise ValueError("At least one mass hypothesis set is required.")
        parsed: list[tuple[float, ...]] = []
        for masses in mass_hypotheses:
            if len(masses) != n_body:
                raise ValueError(
                    f"Mass hypothesis {masses!r} does not match n_body={n_body}."
                )
            parsed.append(tuple(float(x) for x in masses))
        return parsed


# Backward-compatible alias.
TrackCombiner = ParticleCombiner
