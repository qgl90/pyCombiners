"""Multi-sample PID comparison plots."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .performance import PIDConfig, PIDPerformance
from .utils import SAMPLE_COLORS, SAMPLE_MARKERS, plt, require_matplotlib, slug


class PIDComparison:
    """Overlay PID curves from two or more samples / parquet files."""

    def __init__(
        self,
        samples: dict[str, PIDPerformance],
        out_dir: str | Path = ".",
        out_tag: str = "compare",
        colors: Sequence[str] | None = None,
        markers: Sequence[str] | None = None,
    ):
        if len(samples) < 2:
            raise ValueError("PIDComparison needs at least two samples")
        self.samples = samples
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.out_tag = out_tag
        self.colors = list(colors or SAMPLE_COLORS)
        self.markers = list(markers or SAMPLE_MARKERS)
        self.ref = next(iter(samples.values()))

    @classmethod
    def from_parquets(
        cls,
        files: dict[str, str | Path],
        out_dir: str | Path = ".",
        out_tag: str = "compare",
        shared_config: PIDConfig | None = None,
        **common_kwargs,
    ) -> "PIDComparison":
        samples = {}
        for label, path in files.items():
            kwargs = dict(common_kwargs)
            kwargs.setdefault("out_dir", out_dir)
            kwargs.setdefault("out_tag", f"{out_tag}_{slug(label)}")
            kwargs.setdefault("label", label)
            if shared_config is not None:
                kwargs["config"] = shared_config
            samples[label] = PIDPerformance.from_parquet(path, **kwargs)
            print(
                f"[PIDComparison] loaded {label} from {path} with {len(samples[label].df)} tracks"
            )
        return cls(samples, out_dir=out_dir, out_tag=out_tag)

    def plot_roc(self, save: bool = True):
        require_matplotlib()
        fig, ax = plt.subplots(figsize=self.ref.cfg.figsize_roc)
        for i, (label, perf) in enumerate(self.samples.items()):
            perf.plot_roc(
                ax=ax,
                save=False,
                label=label,
                color=self.colors[i % len(self.colors)],
                marker=self.markers[i % len(self.markers)],
            )
        ax.legend(fontsize=9)
        plt.tight_layout()
        if save:
            path = self.out_dir / f"{self.out_tag}_roc_global.png"
            plt.savefig(path, dpi=self.ref.cfg.dpi)
            print(f"Wrote {path}")
        plt.close()
        return ax

    def plot_efficiency_vs_kinematics(
        self,
        var_names: Iterable[str] = ("p", "pt", "eta"),
        targets: Iterable[float] | None = None,
        save: bool = True,
        write_csv: bool = True,
        show_yields: bool = True,
    ) -> dict[str, pd.DataFrame]:
        require_matplotlib()
        targets = tuple(
            self.ref.cfg.default_targets if targets is None else targets
        )
        linestyles = ["-", "--", "-.", ":"]
        out: dict[str, pd.DataFrame] = {}

        for var_name in var_names:
            if var_name not in self.ref.vars:
                print(f"[PID] skip {var_name}: column not available")
                continue
            bins = np.asarray(self.ref.cfg.bins[var_name], dtype=float)
            fig, axes = plt.subplots(
                2 if show_yields else 1,
                1,
                figsize=self.ref.cfg.figsize_kin,
                gridspec_kw={"height_ratios": [3, 1]} if show_yields else None,
                sharex=True,
            )
            ax1 = axes[0] if show_yields else axes
            ax2 = axes[1] if show_yields else None
            combined = []
            for i, (label, perf) in enumerate(self.samples.items()):
                color = self.colors[i % len(self.colors)]
                marker = self.markers[i % len(self.markers)]
                for j, t in enumerate(targets):
                    tab = perf.efficiency_vs_var(
                        var_name, target_misid=t, bins=bins
                    )
                    tab["sample_label"] = label
                    combined.append(tab)
                    ls = linestyles[j % len(linestyles)]
                    lab = (
                        label
                        if len(targets) == 1
                        else f"{label}, mis-ID={t * 100:.0f}%"
                    )
                    ax1.errorbar(
                        tab["center"],
                        tab["efficiency"],
                        yerr=tab["eff_err"],
                        fmt=marker + ls,
                        ms=5,
                        color=color,
                        label=lab,
                    )
                if show_yields:
                    ax2.hist(
                        perf.vars[var_name][perf.mask_sig],
                        bins=bins,
                        histtype="step",
                        color=color,
                        lw=1.6,
                        label=f"true {self.ref._sig_label()} ({label})",
                    )
            misid_txt = (
                rf"{self.ref._bkg_label()}→{self.ref._sig_label()} mis-ID = {targets[0] * 100:.0f}%"
                if len(targets) == 1
                else rf"fixed {self.ref._bkg_label()}→{self.ref._sig_label()} mis-ID"
            )
            ax1.set_ylabel(rf"{self.ref._sig_label()} efficiency")
            ax1.set_ylim(0.0, 1.05)
            ax1.legend(loc="best", fontsize=8)
            ax1.grid(True, alpha=0.3)
            ax1.set_title(
                rf"{self.ref._sig_label()} efficiency at {misid_txt}"
            )
            xlabel = self.ref.cfg.labels.get(var_name, var_name)
            if show_yields:
                ax2.set_xlabel(xlabel)
                ax2.set_ylabel("Candidates")
                ax2.set_yscale("log")
                ax2.legend(fontsize=8)
                ax2.grid(True, alpha=0.3)
            else:
                ax1.set_xlabel(xlabel)
            plt.tight_layout()
            if save:
                path = (
                    self.out_dir
                    / f"{self.out_tag}_eff_vs_{var_name}_fixed_misid.png"
                )
                plt.savefig(path, dpi=self.ref.cfg.dpi)
                print(f"Wrote {path}")
            plt.close()
            stacked = pd.concat(combined, ignore_index=True)
            out[var_name] = stacked
            if write_csv:
                stacked.to_csv(
                    self.out_dir / f"{self.out_tag}_eff_vs_{var_name}.csv",
                    index=False,
                )
        return out

    def plot_efficiency_ratio(
        self,
        var_name: str = "p",
        target_misid: float = 0.05,
        reference: str | None = None,
        save: bool = True,
    ):
        require_matplotlib()
        labels = list(self.samples)
        ref_name = reference or labels[0]
        bins = np.asarray(self.ref.cfg.bins[var_name], dtype=float)
        ref_tab = self.samples[ref_name].efficiency_vs_var(
            var_name, target_misid, bins
        )
        fig, ax = plt.subplots(figsize=self.ref.cfg.figsize_roc)
        for i, (label, perf) in enumerate(self.samples.items()):
            if label == ref_name:
                continue
            tab = perf.efficiency_vs_var(var_name, target_misid, bins)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = (
                    tab["efficiency"].to_numpy()
                    / ref_tab["efficiency"].to_numpy()
                )
            ax.plot(
                tab["center"],
                ratio,
                self.markers[i % len(self.markers)] + "-",
                color=self.colors[i % len(self.colors)],
                label=rf"{label} / {ref_name}",
            )
        ax.axhline(1.0, color="k", lw=1)
        ax.set_xlabel(self.ref.cfg.labels.get(var_name, var_name))
        ax.set_ylabel("efficiency ratio")
        ax.set_title(
            rf"{self.ref._sig_label()} efficiency ratio at "
            rf"{self.ref._bkg_label()}→{self.ref._sig_label()} mis-ID = {target_misid * 100:.0f}%"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        if save:
            path = (
                self.out_dir
                / f"{self.out_tag}_ratio_vs_{var_name}_misid{int(target_misid * 100):02d}.png"
            )
            plt.savefig(path, dpi=self.ref.cfg.dpi)
            print(f"Wrote {path}")
        plt.close()

    def run_all(
        self,
        do_roc: bool = True,
        do_kinematics: bool = True,
        do_ratio: bool = True,
        kin_vars: Sequence[str] = ("p", "pt", "eta"),
        targets: Iterable[float] | None = None,
        ratio_target: float = 0.05,
    ) -> None:
        if do_roc:
            self.plot_roc()
        if do_kinematics:
            self.plot_efficiency_vs_kinematics(
                var_names=kin_vars, targets=targets
            )
        if do_ratio:
            for var in kin_vars:
                if var in self.ref.vars:
                    self.plot_efficiency_ratio(
                        var_name=var, target_misid=ratio_target
                    )
        print("Done. Comparison plots written to", self.out_dir)
