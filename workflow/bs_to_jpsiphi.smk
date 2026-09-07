# Bs -> J/psi phi pipeline: reconstruction + performance/distribution plots.


configfile: "config/bs_to_jpsiphi.yaml"


include: "common.smk"


_RECO = outdir("reconstruction")
_PERF = outdir("analysis", "decay_performance")


wildcard_constraints:
    n_events="[0-9]+",


def _cheated(mode):
    return f"{_RECO}/cheated_{events(f'max_events_{mode}')}.parquet"


def _selection_args():
    values = {
        "max-dt-chi2": config.get("max_dt_chi2", 16.0),
        "min-track-ip-chi2": config.get("min_track_ip_chi2", 9.0),
        "min-jpsi-fd-chi2": config.get("min_jpsi_fd_chi2", 30.0),
        "max-bs-ip-chi2": config.get("max_bs_ip_chi2", 25.0),
        "min-bs-dira": config.get("min_bs_dira", 0.9995),
        "max-jpsi-vertex-chi2": config.get("max_jpsi_vertex_chi2", 30.0),
        "max-phi-vertex-chi2": config.get("max_phi_vertex_chi2", 25.0),
        "max-bs-vertex-chi2": config.get("max_bs_vertex_chi2", 9.0),
    }
    args = [f"--{name} {value}" for name, value in values.items()]
    if not config.get("use_pv_timing", True):
        args.append("--disable-pv-timing")
    if not config.get("require_common_pv_on_time", True):
        args.append("--no-require-common-pv-on-time")
    for particle in ("jpsi", "phi", "bs"):
        value = config.get(f"max_{particle}_vertex_time_chi2")
        if value is not None:
            args.append(f"--max-{particle}-vertex-time-chi2 {value}")
    return " ".join(args)


_SELECTION_ARGS = _selection_args()


rule all:
    input:
        _cheated("cheated"),
        f"{_RECO}/cheated_selected.parquet",
        f"{_RECO}/full.parquet",
        f"{_RECO}/signal_cutflow.parquet",
        f"{_PERF}/bs_eff_vs_kinematics.png",
        f"{_PERF}/bs_mass.png",
        f"{_PERF}/bs_jpsi_mass.png",
        f"{_PERF}/bs_phi_mass.png",
        f"{_PERF}/bs_bkgcat.png",
        f"{_PERF}/bs_bkgcat_yields.parquet",
        f"{_PERF}/bs_bkgcat_yields.csv",
        f"{_PERF}/bs_mass_by_bkgcat.png",
        f"{_PERF}/bs_mass_by_bkgcat_group.png",
        f"{_PERF}/bs_jpsi_mass_by_bkgcat.png",
        f"{_PERF}/bs_phi_mass_by_bkgcat.png",
        f"{_PERF}/bs_selection_efficiency.parquet",
        f"{_PERF}/bs_selection_efficiency.csv",
        f"{_PERF}/bs_selection_cutflow.png",
        f"{_PERF}/bs_selection_cutflow.parquet",
        f"{_PERF}/bs_selection_cutflow.csv",
        outdir("analysis", "truth_distributions") + "/bs_signal_distributions.png",
        outdir("analysis", "reconstructed_distributions") + "/bs_observables.png",


rule jpsiphi_cheated:
    input:
        script="physics/reconstruction/bs_to_jpsiphi.py",
    output:
        f"{_RECO}/cheated_{{n_events}}.parquet",
    log:
        f"{_RECO}/cheated_{{n_events}}.log",
    params:
        data=config["input"],
        selection=_SELECTION_ARGS,
    threads: workflow.cores
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --mode cheated"
        " --input '{params.data}'"
        " --max-events {wildcards.n_events}"
        " --workers {threads}"
        " {params.selection}"
        " --out-file {output}"
        " > {log} 2>&1"


rule jpsiphi_full:
    input:
        cheated=_cheated("full"),
        script="physics/reconstruction/bs_to_jpsiphi.py",
    output:
        f"{_RECO}/full.parquet",
    log:
        f"{_RECO}/full.log",
    params:
        data=config["input"],
        max_events=events("max_events_full"),
        outdir=_RECO,
        selection=_SELECTION_ARGS,
    threads: workflow.cores
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --mode full"
        " --input '{params.data}'"
        " --max-events {params.max_events}"
        " --workers {threads}"
        " {params.selection}"
        " --out-dir {params.outdir}"
        " > {log} 2>&1"


rule jpsiphi_cheated_selected:
    input:
        script="physics/reconstruction/bs_to_jpsiphi.py",
    output:
        f"{_RECO}/cheated_selected.parquet",
    log:
        f"{_RECO}/cheated_selected.log",
    params:
        data=config["input"],
        max_events=events("max_events_full"),
        selection=_SELECTION_ARGS,
    threads: workflow.cores
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --mode cheated_selected"
        " --input '{params.data}'"
        " --max-events {params.max_events}"
        " --workers {threads}"
        " {params.selection}"
        " --out-file {output}"
        " > {log} 2>&1"


rule jpsiphi_signal_cutflow:
    input:
        script="physics/reconstruction/bs_to_jpsiphi.py",
    output:
        f"{_RECO}/signal_cutflow.parquet",
    log:
        f"{_RECO}/signal_cutflow.log",
    params:
        data=config["input"],
        max_events=events("max_events_full"),
        selection=_SELECTION_ARGS,
    threads: workflow.cores
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --mode signal_cutflow"
        " --input '{params.data}'"
        " --max-events {params.max_events}"
        " --workers {threads}"
        " {params.selection}"
        " --out-file {output}"
        " > {log} 2>&1"


rule jpsiphi_decay_performance:
    input:
        cheated=_cheated("full"),
        cheated_selected=f"{_RECO}/cheated_selected.parquet",
        full=f"{_RECO}/full.parquet",
        cutflow=f"{_RECO}/signal_cutflow.parquet",
        script="physics/analysis/decay_performance_study/decay_performance.py",
    output:
        f"{_PERF}/bs_eff_vs_kinematics.png",
        f"{_PERF}/bs_mass.png",
        f"{_PERF}/bs_jpsi_mass.png",
        f"{_PERF}/bs_phi_mass.png",
        f"{_PERF}/bs_bkgcat.png",
        f"{_PERF}/bs_bkgcat_yields.parquet",
        f"{_PERF}/bs_bkgcat_yields.csv",
        f"{_PERF}/bs_mass_by_bkgcat.png",
        f"{_PERF}/bs_mass_by_bkgcat_group.png",
        f"{_PERF}/bs_jpsi_mass_by_bkgcat.png",
        f"{_PERF}/bs_phi_mass_by_bkgcat.png",
        f"{_PERF}/bs_selection_efficiency.parquet",
        f"{_PERF}/bs_selection_efficiency.csv",
        f"{_PERF}/bs_selection_cutflow.png",
        f"{_PERF}/bs_selection_cutflow.parquet",
        f"{_PERF}/bs_selection_cutflow.csv",
    log:
        f"{_PERF}/run.log",
    params:
        outdir=_PERF,
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --cheated {input.cheated}"
        " --cheated-selected {input.cheated_selected}"
        " --full {input.full}"
        " --cutflow {input.cutflow}"
        " --channel bs_to_jpsiphi"
        " --out-dir {params.outdir}"
        " > {log} 2>&1"


rule jpsiphi_truth_distributions:
    input:
        cheated=_cheated("cheated"),
        script=("physics/analysis/truth_distribution_study/signal_distributions.py"),
    output:
        outdir("analysis", "truth_distributions") + "/bs_signal_distributions.png",
    log:
        outdir("analysis", "truth_distributions") + "/run.log",
    params:
        outdir=outdir("analysis", "truth_distributions"),
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --cheated {input.cheated}"
        " --channel bs_to_jpsiphi"
        " --out-dir {params.outdir}"
        " > {log} 2>&1"


rule jpsiphi_observables:
    input:
        dist=f"{_RECO}/full.parquet",
        script=(
            "physics/analysis/reconstructed_distribution_study/"
            "signal_vs_background.py"
        ),
    output:
        outdir("analysis", "reconstructed_distributions") + "/bs_observables.png",
    log:
        outdir("analysis", "reconstructed_distributions") + "/run.log",
    params:
        outdir=outdir("analysis", "reconstructed_distributions"),
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --dist {input.dist}"
        " --channel bs_to_jpsiphi"
        " --out-dir {params.outdir}"
        " > {log} 2>&1"
