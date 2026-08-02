# Bs -> J/psi phi pipeline: reconstruction + performance/distribution plots.


configfile: "config/bs_to_jpsiphi.yaml"


include: "common.smk"


_RECO = outdir("reconstruction")
_PERF = outdir("analysis", "decay_performance")


wildcard_constraints:
    n_events="[0-9]+",


def _cheated(mode):
    return f"{_RECO}/cheated_{events(f'max_events_{mode}')}.parquet"


rule all:
    input:
        _cheated("cheated"),
        f"{_RECO}/full.parquet",
        f"{_PERF}/bs_eff_vs_kinematics.png",
        f"{_PERF}/bs_mass.png",
        f"{_PERF}/bs_jpsi_mass.png",
        f"{_PERF}/bs_phi_mass.png",
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
    threads: workflow.cores
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --mode cheated"
        " --input '{params.data}'"
        " --max-events {wildcards.n_events}"
        " --workers {threads}"
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
    threads: workflow.cores
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --mode full"
        " --input '{params.data}'"
        " --max-events {params.max_events}"
        " --workers {threads}"
        " --out-dir {params.outdir}"
        " > {log} 2>&1"


rule jpsiphi_decay_performance:
    input:
        cheated=_cheated("full"),
        full=f"{_RECO}/full.parquet",
        script="physics/analysis/decay_performance_study/decay_performance.py",
    output:
        f"{_PERF}/bs_eff_vs_kinematics.png",
        f"{_PERF}/bs_mass.png",
        f"{_PERF}/bs_jpsi_mass.png",
        f"{_PERF}/bs_phi_mass.png",
    log:
        f"{_PERF}/run.log",
    params:
        outdir=_PERF,
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --cheated {input.cheated} --full {input.full}"
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
