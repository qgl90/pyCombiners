"""Single-sample PID performance analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .utils import (
    DEFAULT_BINS,
    DEFAULT_COLORS,
    DEFAULT_LABELS,
    LogNorm,
    binom_err,
    plt,
    require_matplotlib,
    resolve_pdg_id,
    species_label,
)


@dataclass
class PIDConfig:
    """Tunable knobs for PID performance studies."""

    signal_id: str | int = "K+"
    background_id: str | int = "pi+"
    dll_field: str = "rich_dll_kaon"
    require_rich: bool = True
    require_truth: bool = True
    higher_dll_is_signal: bool = True

    truth_field: str = "mc_truth"
    mc_pid_field: str = "mc_pid"
    rich_flag_field: str = "rich_has_info"

    eta_field: str = "eta"
    pt_field: str = "pt"
    p_field: str = "p"

    auto_scale_mev: bool = False
    p_mev_threshold: float = 500.0

    bins: dict[str, Sequence[float]] = field(
        default_factory=lambda: dict(DEFAULT_BINS)
    )
    labels: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_LABELS)
    )

    n_scan_points: int = 80
    dll_percentile: tuple[float, float] = (0.5, 99.5)

    roc_signal_axis: str = "x"
    roc_log_misid: bool = True

    default_targets: tuple[float, ...] = (0.03, 0.05, 0.10, 0.15)
    wp_signal_efficiency: float = 0.90

    min_signal_per_bin: int = 10
    min_background_per_bin: int = 10

    dpi: int = 150
    figsize_roc: tuple[float, float] = (7.0, 5.0)
    figsize_kin: tuple[float, float] = (8.0, 7.0)


class PIDPerformance:
    """Analyse RICH (or other) DLL PID performance from a track DataFrame."""

    def __init__(
        self,
        df: pd.DataFrame,
        out_dir: str | Path = ".",
        out_tag: str = "pid_performance",
        pdg_id_fn: Callable | None = None,
        config: PIDConfig | None = None,
        label: str | None = None,
        **config_overrides,
    ):
        self.df = df
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.out_tag = out_tag
        self.label = label or out_tag
        self.pdg_id_fn = pdg_id_fn

        cfg = config or PIDConfig()
        for key, value in config_overrides.items():
            if not hasattr(cfg, key):
                raise AttributeError(f"Unknown PIDConfig field: {key}")
            setattr(cfg, key, value)
        self.cfg = cfg

        self.signal_pdg = abs(resolve_pdg_id(cfg.signal_id, pdg_id_fn))
        self.background_pdg = abs(resolve_pdg_id(cfg.background_id, pdg_id_fn))
        self._prepare()

    @classmethod
    def from_parquet(cls, path: str | Path, **kwargs) -> "PIDPerformance":
        return cls(pd.read_parquet(path), **kwargs)

    def _prepare(self) -> None:
        cfg = self.cfg
        df = self.df
        truth = (
            df[cfg.truth_field].astype(bool).to_numpy()
            if cfg.require_truth and cfg.truth_field in df.columns
            else np.ones(len(df), dtype=bool)
        )
        has_rich = (
            df[cfg.rich_flag_field].astype(bool).to_numpy()
            if cfg.require_rich and cfg.rich_flag_field in df.columns
            else np.ones(len(df), dtype=bool)
        )
        mc_pid = np.abs(df[cfg.mc_pid_field].to_numpy())
        self.mask_sig = truth & has_rich & (mc_pid == self.signal_pdg)
        self.mask_bkg = truth & has_rich & (mc_pid == self.background_pdg)
        self.dll = df[cfg.dll_field].to_numpy(dtype=float)

        self.vars: dict[str, np.ndarray] = {}
        for name, col in (
            ("p", cfg.p_field),
            ("pt", cfg.pt_field),
            ("eta", cfg.eta_field),
        ):
            if col in df.columns:
                values = df[col].to_numpy(dtype=float)
                if name in ("p", "pt") and cfg.auto_scale_mev:
                    sample = values[np.isfinite(values)]
                    if (
                        sample.size
                        and np.nanmedian(np.abs(sample)) > cfg.p_mev_threshold
                    ):
                        values = values / 1000.0
                self.vars[name] = values

        self.n_sig = int(self.mask_sig.sum())
        self.n_bkg = int(self.mask_bkg.sum())
        print(
            f"[PID] {self.label}: signal={cfg.signal_id} ({self.n_sig})  "
            f"background={cfg.background_id} ({self.n_bkg})  dll={cfg.dll_field}"
        )

    def scan_points(self, mask_extra: np.ndarray | None = None) -> np.ndarray:
        mask = self.mask_sig | self.mask_bkg
        if mask_extra is not None:
            mask = mask & mask_extra
        sample = self.dll[mask & np.isfinite(self.dll)]
        if sample.size == 0:
            return np.linspace(-10.0, 10.0, self.cfg.n_scan_points)
        lo, hi = np.percentile(sample, self.cfg.dll_percentile)
        if lo == hi:
            lo, hi = lo - 1.0, hi + 1.0
        return np.linspace(lo, hi, self.cfg.n_scan_points)

    def scan_cut(
        self,
        mask_sig: np.ndarray | None = None,
        mask_bkg: np.ndarray | None = None,
        cuts: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        mask_sig = self.mask_sig if mask_sig is None else mask_sig
        mask_bkg = self.mask_bkg if mask_bkg is None else mask_bkg
        cuts = (
            self.scan_points()
            if cuts is None
            else np.asarray(cuts, dtype=float)
        )
        n_sig = int(mask_sig.sum())
        n_bkg = int(mask_bkg.sum())
        if n_sig == 0 or n_bkg == 0:
            nan = np.full_like(cuts, np.nan, dtype=float)
            return nan, nan

        dll = self.dll
        finite = np.isfinite(dll)
        mask_sig = mask_sig & finite
        mask_bkg = mask_bkg & finite
        n_sig = int(mask_sig.sum())
        n_bkg = int(mask_bkg.sum())
        if n_sig == 0 or n_bkg == 0:
            nan = np.full_like(cuts, np.nan, dtype=float)
            return nan, nan

        order = np.argsort(dll, kind="mergesort")
        dll_sorted = dll[order]
        sig_cum = np.concatenate(([0], np.cumsum(mask_sig[order])))
        bkg_cum = np.concatenate(([0], np.cumsum(mask_bkg[order])))
        idx = np.searchsorted(dll_sorted, cuts, side="left")
        if self.cfg.higher_dll_is_signal:
            return (n_sig - sig_cum[idx]) / n_sig, (
                n_bkg - bkg_cum[idx]
            ) / n_bkg
        return sig_cum[idx] / n_sig, bkg_cum[idx] / n_bkg

    def global_roc(
        self, cuts: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        cuts = self.scan_points() if cuts is None else cuts
        eff, misid = self.scan_cut(cuts=cuts)
        return {"cuts": cuts, "efficiency": eff, "misid": misid}

    def binned_roc(
        self,
        var_name: str,
        bins: Sequence[float] | None = None,
        cuts: np.ndarray | None = None,
    ) -> dict[float, dict[str, np.ndarray]]:
        var = self.vars[var_name]
        bins = np.asarray(
            bins if bins is not None else self.cfg.bins[var_name], dtype=float
        )
        cuts = self.scan_points() if cuts is None else cuts
        out = {}
        for lo, hi in zip(bins[:-1], bins[1:]):
            in_bin = (var >= lo) & (var < hi)
            eff, misid = self.scan_cut(
                self.mask_sig & in_bin, self.mask_bkg & in_bin, cuts=cuts
            )
            out[0.5 * (lo + hi)] = {
                "cuts": cuts,
                "efficiency": eff,
                "misid": misid,
            }
        return out

    def cut_for_misid(
        self, mask_bkg: np.ndarray | None = None, target_misid: float = 0.05
    ) -> float:
        mask_bkg = self.mask_bkg if mask_bkg is None else mask_bkg
        sample = self.dll[mask_bkg & np.isfinite(self.dll)]
        if sample.size == 0:
            return np.nan
        q = (
            1.0 - target_misid
            if self.cfg.higher_dll_is_signal
            else target_misid
        )
        return float(np.quantile(sample, min(max(q, 0.0), 1.0)))

    def efficiency_at_cut(
        self, mask_sig: np.ndarray | None = None, cut: float = 0.0
    ) -> float:
        mask_sig = self.mask_sig if mask_sig is None else mask_sig
        if mask_sig.sum() == 0 or not np.isfinite(cut):
            return np.nan
        dll = self.dll[mask_sig]
        passed = dll > cut if self.cfg.higher_dll_is_signal else dll < cut
        return float(np.mean(passed))

    def efficiency_vs_var(
        self,
        var_name: str,
        target_misid: float = 0.05,
        bins: Sequence[float] | None = None,
    ) -> pd.DataFrame:
        var = self.vars[var_name]
        bins = np.asarray(
            bins if bins is not None else self.cfg.bins[var_name], dtype=float
        )
        centers = 0.5 * (bins[:-1] + bins[1:])
        rows = []
        for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
            in_bin = (var >= lo) & (var < hi)
            sig_bin = self.mask_sig & in_bin
            bkg_bin = self.mask_bkg & in_bin
            n_sig = int(sig_bin.sum())
            n_bkg = int(bkg_bin.sum())
            if (
                n_sig < self.cfg.min_signal_per_bin
                or n_bkg < self.cfg.min_background_per_bin
            ):
                cut = eff = misid_obs = np.nan
            else:
                cut = self.cut_for_misid(bkg_bin, target_misid)
                eff = self.efficiency_at_cut(sig_bin, cut)
                misid_obs = self.efficiency_at_cut(bkg_bin, cut)
            rows.append(
                {
                    "sample": self.label,
                    "var": var_name,
                    "bin_lo": lo,
                    "bin_hi": hi,
                    "center": centers[i],
                    "target_misid": target_misid,
                    "efficiency": eff,
                    "misid_observed": misid_obs,
                    "n_signal": n_sig,
                    "n_background": n_bkg,
                    "dll_cut": cut,
                    "eff_err": binom_err(eff, n_sig),
                }
            )
        return pd.DataFrame(rows)

    def map_2d(
        self,
        cut: float | None = None,
        x_name: str = "eta",
        y_name: str = "pt",
        x_bins: Sequence[float] | None = None,
        y_bins: Sequence[float] | None = None,
    ) -> dict[str, np.ndarray]:
        if cut is None:
            roc = self.global_roc()
            idx = int(
                np.nanargmin(
                    np.abs(roc["efficiency"] - self.cfg.wp_signal_efficiency)
                )
            )
            cut = float(roc["cuts"][idx])
        x = self.vars[x_name]
        y = self.vars[y_name]
        x_bins = np.asarray(
            x_bins if x_bins is not None else self.cfg.bins[x_name],
            dtype=float,
        )
        y_bins = np.asarray(
            y_bins if y_bins is not None else self.cfg.bins[y_name],
            dtype=float,
        )
        passed = (
            (self.dll > cut)
            if self.cfg.higher_dll_is_signal
            else (self.dll < cut)
        )

        def _rate(mask):
            h_all, _, _ = np.histogram2d(
                x[mask], y[mask], bins=[x_bins, y_bins]
            )
            h_pass, _, _ = np.histogram2d(
                x[mask & passed], y[mask & passed], bins=[x_bins, y_bins]
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                return h_pass / h_all, h_all

        eff, n_sig = _rate(self.mask_sig)
        misid, n_bkg = _rate(self.mask_bkg)
        return {
            "cut": np.array(cut),
            "x_bins": x_bins,
            "y_bins": y_bins,
            "efficiency": eff,
            "misid": misid,
            "n_signal": n_sig,
            "n_background": n_bkg,
        }

    def _sig_label(self) -> str:
        return species_label(self.cfg.signal_id)

    def _bkg_label(self) -> str:
        return species_label(self.cfg.background_id)

    def plot_roc(
        self,
        var_name: str | None = None,
        bins: Sequence[float] | None = None,
        ax=None,
        save: bool = True,
        label: str | None = None,
        color: str | None = None,
        marker: str = "o",
    ):
        require_matplotlib()
        cuts = self.scan_points()
        created = ax is None
        if created:
            _, ax = plt.subplots(figsize=self.cfg.figsize_roc)

        def _draw(eff, misid, lab):
            xx, yy = (
                (eff, misid)
                if self.cfg.roc_signal_axis == "x"
                else (misid, eff)
            )
            ax.plot(xx, yy, marker + "-", ms=3, color=color, label=lab)

        lab = label or self.label
        if var_name is None:
            roc = self.global_roc(cuts)
            _draw(roc["efficiency"], roc["misid"], lab)
            title = f"{self.cfg.dll_field}  {self._sig_label()} vs {self._bkg_label()}"
            tag = "roc_global"
            n_curves = 1
        else:
            curves = self.binned_roc(var_name, bins=bins, cuts=cuts)
            unit = self.cfg.labels.get(var_name, var_name)
            for center, roc in curves.items():
                _draw(
                    roc["efficiency"],
                    roc["misid"],
                    f"{var_name} ≈ {center:.2f}",
                )
            title = f"vs {unit}"
            tag = f"roc_vs_{var_name}"
            n_curves = len(curves)

        if self.cfg.roc_signal_axis == "x":
            ax.set_xlabel(f"{self._sig_label()} efficiency")
            ax.set_ylabel(f"{self._bkg_label()} → {self._sig_label()} mis-ID")
            if self.cfg.roc_log_misid:
                ax.set_yscale("log")
        else:
            ax.set_ylabel(f"{self._sig_label()} efficiency")
            ax.set_xlabel(f"{self._bkg_label()} → {self._sig_label()} mis-ID")
            if self.cfg.roc_log_misid:
                ax.set_xscale("log")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        if n_curves <= 12:
            ax.legend(fontsize=8)
        if created:
            plt.tight_layout()
            if save:
                path = self.out_dir / f"{self.out_tag}_{tag}.png"
                plt.savefig(path, dpi=self.cfg.dpi)
                print(f"Wrote {path}")
            plt.close()
        return ax

    def plot_efficiency_vs_kinematics(
        self,
        var_names: Iterable[str] = ("p", "pt", "eta"),
        targets: Iterable[float] | None = None,
        save: bool = True,
        write_csv: bool = True,
    ) -> dict[str, dict[float, pd.DataFrame]]:
        require_matplotlib()
        targets = tuple(
            self.cfg.default_targets if targets is None else targets
        )
        all_tables: dict[str, dict[float, pd.DataFrame]] = {}
        for var_name in var_names:
            if var_name not in self.vars:
                print(f"[PID] skip {var_name}: column not available")
                continue
            bins = np.asarray(self.cfg.bins[var_name], dtype=float)
            tables = {
                t: self.efficiency_vs_var(var_name, target_misid=t, bins=bins)
                for t in targets
            }
            all_tables[var_name] = tables
            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                figsize=self.cfg.figsize_kin,
                gridspec_kw={"height_ratios": [3, 1]},
                sharex=True,
            )
            for t, tab in tables.items():
                ax1.errorbar(
                    tab["center"],
                    tab["efficiency"],
                    yerr=tab["eff_err"],
                    fmt="o-",
                    ms=5,
                    color=DEFAULT_COLORS.get(t),
                    label=rf"{self._bkg_label()}→{self._sig_label()} mis-ID = {t * 100:.0f}%",
                )
            ax1.set_ylabel(rf"{self._sig_label()} efficiency")
            ax1.set_ylim(0.0, 1.05)
            ax1.legend(loc="best", fontsize=9)
            ax1.grid(True, alpha=0.3)
            ax1.set_title(
                rf"{self._sig_label()} efficiency at fixed "
                rf"{self._bkg_label()}→{self._sig_label()} mis-ID"
            )
            ax2.hist(
                self.vars[var_name][self.mask_sig],
                bins=bins,
                histtype="stepfilled",
                color="0.7",
                edgecolor="k",
                alpha=0.8,
                label=f"true {self._sig_label()}",
            )
            ax2.set_xlabel(self.cfg.labels.get(var_name, var_name))
            ax2.set_ylabel("Candidates")
            ax2.set_yscale("log")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            if save:
                path = (
                    self.out_dir
                    / f"{self.out_tag}_eff_vs_{var_name}_fixed_misid.png"
                )
                plt.savefig(path, dpi=self.cfg.dpi)
                print(f"Wrote {path}")
            plt.close()
            if write_csv:
                for t, tab in tables.items():
                    tab.to_csv(
                        self.out_dir
                        / f"{self.out_tag}_eff_vs_{var_name}_misid{int(t * 100):02d}.csv",
                        index=False,
                    )
        return all_tables

    def plot_maps_2d(self, cut: float | None = None, save: bool = True):
        require_matplotlib()
        maps = self.map_2d(cut=cut)
        cut = float(maps["cut"])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        im1 = ax1.pcolormesh(
            maps["x_bins"],
            maps["y_bins"],
            maps["efficiency"].T,
            shading="auto",
            vmin=0,
            vmax=1,
        )
        ax1.set_xlabel(self.cfg.labels.get("eta", "η"))
        ax1.set_ylabel(self.cfg.labels.get("pt", "pT"))
        ax1.set_title(f"{self._sig_label()} efficiency (DLL cut {cut:.2f})")
        fig.colorbar(im1, ax=ax1)
        im2 = ax2.pcolormesh(
            maps["x_bins"],
            maps["y_bins"],
            maps["misid"].T,
            shading="auto",
            norm=LogNorm(vmin=1e-4, vmax=1),
        )
        ax2.set_xlabel(self.cfg.labels.get("eta", "η"))
        ax2.set_ylabel(self.cfg.labels.get("pt", "pT"))
        ax2.set_title(
            f"{self._bkg_label()}→{self._sig_label()} mis-ID (DLL cut {cut:.2f})"
        )
        fig.colorbar(im2, ax=ax2)
        plt.tight_layout()
        if save:
            path = self.out_dir / f"{self.out_tag}_maps_2d.png"
            plt.savefig(path, dpi=self.cfg.dpi)
            print(f"Wrote {path}")
        plt.close()
        return maps

    def run_all(
        self,
        do_roc: bool = True,
        do_kinematics: bool = True,
        do_maps: bool = True,
        roc_vars: Sequence[str] = ("eta", "pt", "p"),
        kin_vars: Sequence[str] = ("p", "pt", "eta"),
        targets: Iterable[float] | None = None,
    ) -> None:
        if do_roc:
            self.plot_roc()
            for var in roc_vars:
                if var in self.vars:
                    self.plot_roc(var_name=var)
        if do_kinematics:
            self.plot_efficiency_vs_kinematics(
                var_names=kin_vars, targets=targets
            )
        if do_maps and "eta" in self.vars and "pt" in self.vars:
            self.plot_maps_2d()
        print("Done. Plots written to", self.out_dir)
