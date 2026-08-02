# Ks -> pi+ pi- pipeline: reconstruction + performance/distribution plots.


configfile: "config/ks_to_pipi.yaml"


include: "common.smk"


_RECO = outdir("reconstruction")
_RECO_MODES = ["full", "dist"]


wildcard_constraints:
    ks_mode="full|dist",
    n_events="[0-9]+",


def _cheated(mode):
    return f"{_RECO}/cheated_{events(f'max_events_{mode}')}.parquet"


rule all:
    input:
        _cheated("cheated"),
        expand(f"{_RECO}/{{mode}}.parquet", mode=_RECO_MODES),
        outdir("analysis", "decay_performance") + "/ks_eff_vs_kinematics.png",
        outdir("analysis", "decay_performance") + "/ks_mass.png",
        outdir("analysis", "truth_distributions") + "/ks_signal_distributions.png",
        outdir("analysis", "reconstructed_distributions") + "/ks_observables.png",


rule ks_cheated:
    input:
        script="physics/reconstruction/ks_to_pipi.py",
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


rule ks_reco:
    input:
        cheated=lambda wc: _cheated(wc.ks_mode),
        script="physics/reconstruction/ks_to_pipi.py",
    output:
        f"{_RECO}/{{ks_mode}}.parquet",
    log:
        f"{_RECO}/{{ks_mode}}.log",
    params:
        data=config["input"],
        max_events=lambda wc: events(f"max_events_{wc.ks_mode}"),
        outdir=_RECO,
    threads: workflow.cores
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --mode {wildcards.ks_mode}"
        " --input '{params.data}'"
        " --max-events {params.max_events}"
        " --workers {threads}"
        " --out-dir {params.outdir}"
        " > {log} 2>&1"


rule ks_decay_performance:
    input:
        cheated=_cheated("full"),
        full=f"{_RECO}/full.parquet",
        script="physics/analysis/decay_performance_study/decay_performance.py",
    output:
        outdir("analysis", "decay_performance") + "/ks_eff_vs_kinematics.png",
        outdir("analysis", "decay_performance") + "/ks_mass.png",
    log:
        outdir("analysis", "decay_performance") + "/run.log",
    params:
        outdir=outdir("analysis", "decay_performance"),
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --cheated {input.cheated} --full {input.full}"
        " --channel ks_to_pipi"
        " --out-dir {params.outdir}"
        " > {log} 2>&1"


rule ks_truth_distributions:
    input:
        cheated=_cheated("cheated"),
        script=("physics/analysis/truth_distribution_study/signal_distributions.py"),
    output:
        outdir("analysis", "truth_distributions") + "/ks_signal_distributions.png",
    log:
        outdir("analysis", "truth_distributions") + "/run.log",
    params:
        outdir=outdir("analysis", "truth_distributions"),
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --cheated {input.cheated}"
        " --channel ks_to_pipi"
        " --out-dir {params.outdir}"
        " > {log} 2>&1"


rule ks_observables:
    input:
        dist=f"{_RECO}/dist.parquet",
        script=(
            "physics/analysis/reconstructed_distribution_study/"
            "signal_vs_background.py"
        ),
    output:
        outdir("analysis", "reconstructed_distributions") + "/ks_observables.png",
    log:
        outdir("analysis", "reconstructed_distributions") + "/run.log",
    params:
        outdir=outdir("analysis", "reconstructed_distributions"),
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --dist {input.dist}"
        " --channel ks_to_pipi"
        " --out-dir {params.outdir}"
        " > {log} 2>&1"
