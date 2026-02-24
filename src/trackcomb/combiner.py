"""High-level combination engine for event tracks and PV hypotheses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .models import (
    CombinationCuts,
    CombinationResult,
    EventInput,
    ParticleHypothesis,
    PrimaryVertex,
    TrackPreselection,
    TrackState,
    iter_n_body_combinations,
)
from .physics import (
    C_LIGHT_MM_PER_NS,
    fit_vertex_time,
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

    speed_of_light: float = C_LIGHT_MM_PER_NS

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
        mass_hypotheses: Sequence[Sequence[float | ParticleHypothesis]],
        preselection: TrackPreselection | None = None,
        cuts: CombinationCuts | None = None,
        event_id: str | None = None,
    ) -> list[CombinationResult]:
        """Build n-body candidates and apply candidate-level cuts.

        Workflow:
        1. Preselect tracks.
        2. Enumerate n-body combinations.
        3. Fit spatial vertex `(x,y,z)` from track geometry.
        4. For each mass assignment, propagate track times with beta correction.
        5. Apply cuts (doca, chi2, mass, pt, eta, charge pattern).
        6. Return accepted `CombinationResult` objects.
        """
        selected_tracks = self.preselect_tracks(tracks, primary_vertices, preselection)
        if not selected_tracks:
            return []
        pvs = list(primary_vertices)
        if not pvs:
            raise ValueError("At least one primary vertex is required.")
        valid_hypotheses = self._validate_hypotheses(mass_hypotheses, n_body)
        cuts = cuts or CombinationCuts()
        self._validate_charge_patterns(cuts.allowed_charge_patterns, n_body)

        results: list[CombinationResult] = []
        for combo in iter_n_body_combinations(selected_tracks, n_body):
            combo_tracks = list(combo)
            # Geometry-only fit: the vertex position is solved once per track tuple.
            fit = fit_vertex_xyz_t(
                combo_tracks,
                masses=None,
                speed_of_light=self.speed_of_light,
            )
            if cuts.max_vertex_chi2 is not None and fit.spatial_chi2 > cuts.max_vertex_chi2:
                continue

            doca_pairs = pairwise_doca(combo_tracks)
            if cuts.max_doca is not None and any(v > cuts.max_doca for v in doca_pairs.values()):
                continue

            vertices_xy = tuple(t.extrapolate(fit.vertex_xyz[2]) for t in combo_tracks)
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
            source_track_ids: list[str] = []
            for t in combo_tracks:
                if t.source_track_ids:
                    source_track_ids.extend(t.source_track_ids)
                else:
                    source_track_ids.append(t.track_id)
            source_track_ids = list(dict.fromkeys(source_track_ids))

            for hypotheses in valid_hypotheses:
                # Timing compatibility is mass-dependent through beta, so this is
                # evaluated for each hypothesis assignment separately.
                masses = tuple(h.mass for h in hypotheses)
                time_fit = fit_vertex_time(
                    tracks=combo_tracks,
                    masses=masses,
                    vertex_xyz=fit.vertex_xyz,
                    speed_of_light=self.speed_of_light,
                )
                pair_time = pairwise_time_chi2(
                    combo_tracks,
                    masses=masses,
                    vertex_xyz=fit.vertex_xyz,
                    speed_of_light=self.speed_of_light,
                )
                if cuts.max_vertex_time_chi2 is not None and time_fit.chi2 > cuts.max_vertex_time_chi2:
                    continue
                if cuts.max_pair_time_chi2 is not None and pair_time > cuts.max_pair_time_chi2:
                    continue

                # Candidate four-momentum always follows the active hypothesis tuple.
                p4 = sum_lorentz(
                    track_to_lorentz(track, hypothesis.mass)
                    for track, hypothesis in zip(combo_tracks, hypotheses, strict=True)
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
                        masses=masses,
                        particle_hypotheses=tuple(h.name for h in hypotheses),
                        vertex_xyz=fit.vertex_xyz,
                        vertex_cov_xyz=fit.cov_xyz,
                        vertex_time=time_fit.vertex_time,
                        vertex_sigma_time=time_fit.sigma_time,
                        vertices_xy=vertices_xy,
                        candidate_p4=p4,
                        vertex_chi2=fit.spatial_chi2,
                        vertex_time_chi2=time_fit.chi2,
                        pair_time_chi2=pair_time,
                        doca_pairs=doca_pairs,
                        track_min_ip=track_min_ip,
                        track_min_ip_chi2=track_min_ip_chi2,
                        track_charges=track_charges,
                        track_pid_info=track_pid_info,
                        charge_pattern=charge_pattern,
                        total_charge=total_charge,
                        pair_pt=pair_pt,
                        pair_eta=pair_eta,
                        source_track_ids=tuple(source_track_ids),
                        event_id=event_id,
                        best_pv_id=best_pv_id,
                    )
                )
        return results

    def combine_events(
        self,
        events: Sequence[EventInput],
        n_body: int,
        mass_hypotheses: Sequence[Sequence[float | ParticleHypothesis]],
        preselection: TrackPreselection | None = None,
        cuts: CombinationCuts | None = None,
    ) -> list[CombinationResult]:
        """Run `combine` on a list of events and aggregate tagged candidates."""
        out: list[CombinationResult] = []
        for event in events:
            out.extend(
                self.combine(
                    tracks=event.tracks,
                    primary_vertices=event.primary_vertices,
                    n_body=n_body,
                    mass_hypotheses=mass_hypotheses,
                    preselection=preselection,
                    cuts=cuts,
                    event_id=event.event_id,
                )
            )
        return out

    @staticmethod
    def _validate_hypotheses(
        mass_hypotheses: Sequence[Sequence[float | ParticleHypothesis]],
        n_body: int,
    ) -> list[tuple[ParticleHypothesis, ...]]:
        """Validate hypothesis shape and coerce floats into named hypotheses."""
        if not mass_hypotheses:
            raise ValueError("At least one mass hypothesis set is required.")
        parsed: list[tuple[ParticleHypothesis, ...]] = []
        for hyp_set in mass_hypotheses:
            if len(hyp_set) != n_body:
                raise ValueError(
                    f"Mass hypothesis {hyp_set!r} does not match n_body={n_body}."
                )
            parsed_set: list[ParticleHypothesis] = []
            for item in hyp_set:
                if isinstance(item, ParticleHypothesis):
                    parsed_set.append(item)
                else:
                    mass = float(item)
                    parsed_set.append(ParticleHypothesis(name=f"m={mass:g}", mass=mass))
            parsed.append(tuple(parsed_set))
        return parsed

    @staticmethod
    def _validate_charge_patterns(
        allowed_charge_patterns: tuple[str, ...] | None,
        n_body: int,
    ) -> None:
        """Validate optional charge-pattern filters against current n-body mode."""
        if allowed_charge_patterns is None:
            return
        for pat in allowed_charge_patterns:
            if len(pat) != n_body:
                raise ValueError(
                    f"Charge pattern '{pat}' length does not match n_body={n_body}."
                )
            if any(ch not in "+-0" for ch in pat):
                raise ValueError(
                    f"Charge pattern '{pat}' contains invalid symbol. Use +, -, or 0."
                )


# Backward-compatible alias.
TrackCombiner = ParticleCombiner
