# Bs -> phi gamma pipeline: cheated reconstruction + mass plots.


configfile: "config/bs_to_phigamma.yaml"


include: "common.smk"


_RECO = outdir("reconstruction")
_PERF = outdir("analysis")


rule all:
    input:
        f"{_PERF}/phi_mass.png",
        f"{_PERF}/bs_mass.png",
        f"{_PERF}/mass_rawE.png",


rule phigamma_cheated:
    input:
        script="physics/reconstruction/bs_to_phigamma.py",
    output:
        f"{_RECO}/cheated.parquet",
    log:
        f"{_RECO}/cheated.log",
    params:
        data=config["input"],
        max_events=config["max_events"],
    threads: workflow.cores
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --input '{params.data}'"
        " --max-events {params.max_events}"
        " --workers {threads}"
        " --out-file {output}"
        " > {log} 2>&1"


rule phigamma_cheated_plots:
    input:
        parquet=f"{_RECO}/cheated.parquet",
        script=("physics/analysis/decay_performance_study/bs_to_phigamma_cheated.py"),
    output:
        f"{_PERF}/phi_mass.png",
        f"{_PERF}/bs_mass.png",
        f"{_PERF}/mass_rawE.png",
    log:
        f"{_PERF}/run.log",
    params:
        outdir=_PERF,
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --input {input.parquet}"
        " --out-dir {params.outdir}"
        " > {log} 2>&1"
