_OUTDIR = config["output_dir"]
_LUMIS = ["1p5e34"]
_SCRIPT = "physics/analysis/reconstructed_distribution_study/signal_vs_background.py"


rule all_reconstructed_distribution_study:
    input:
        expand(
            f"{_OUTDIR}/{{lumi}}/reconstructed_distribution_study/ks_to_pipi/ks_observables.png",
            lumi=_LUMIS,
        ),
        expand(
            f"{_OUTDIR}/{{lumi}}/reconstructed_distribution_study/bs_to_mumu/bs_observables.png",
            lumi=_LUMIS,
        ),
        expand(
            f"{_OUTDIR}/{{lumi}}/reconstructed_distribution_study/bs_to_jpsiphi/bs_observables.png",
            lumi=_LUMIS,
        ),


rule reconstructed_distribution_ks_to_pipi:
    input:
        dist=f"{_OUTDIR}/{{lumi}}/reconstruction/ks_to_pipi/dist.parquet",
        script=_SCRIPT,
    output:
        f"{_OUTDIR}/{{lumi}}/reconstructed_distribution_study/ks_to_pipi/ks_observables.png",
    log:
        f"{_OUTDIR}/{{lumi}}/reconstructed_distribution_study/ks_to_pipi/run.log",
    params:
        outdir=_OUTDIR,
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --dist {input.dist}"
        " --channel ks_to_pipi"
        " --out-dir {params.outdir}/{wildcards.lumi}/reconstructed_distribution_study/ks_to_pipi"
        " --lumi {wildcards.lumi}"
        " > {log} 2>&1"


rule reconstructed_distribution_bs_to_mumu:
    input:
        dist=f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/dist.parquet",
        script=_SCRIPT,
    output:
        f"{_OUTDIR}/{{lumi}}/reconstructed_distribution_study/bs_to_mumu/bs_observables.png",
    log:
        f"{_OUTDIR}/{{lumi}}/reconstructed_distribution_study/bs_to_mumu/run.log",
    params:
        outdir=_OUTDIR,
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --dist {input.dist}"
        " --channel bs_to_mumu"
        " --out-dir {params.outdir}/{wildcards.lumi}/reconstructed_distribution_study/bs_to_mumu"
        " --lumi {wildcards.lumi}"
        " > {log} 2>&1"


rule reconstructed_distribution_bs_to_jpsiphi:
    input:
        dist=f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_jpsiphi/full.parquet",
        script=_SCRIPT,
    output:
        f"{_OUTDIR}/{{lumi}}/reconstructed_distribution_study/bs_to_jpsiphi/bs_observables.png",
    log:
        f"{_OUTDIR}/{{lumi}}/reconstructed_distribution_study/bs_to_jpsiphi/run.log",
    params:
        outdir=_OUTDIR,
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --dist {input.dist}"
        " --channel bs_to_jpsiphi"
        " --out-dir {params.outdir}/{wildcards.lumi}/reconstructed_distribution_study/bs_to_jpsiphi"
        " --lumi {wildcards.lumi}"
        " > {log} 2>&1"
