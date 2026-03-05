_OUTDIR = config["output_dir"]
_LUMIS = ["1p5e34"]


rule all_reconstructed_distribution_study:
    input:
        expand(f"{_OUTDIR}/{{lumi}}/reconstructed_distribution_study/ks_to_pipi/ks_observables.png",
               lumi=_LUMIS),
        expand(f"{_OUTDIR}/{{lumi}}/reconstructed_distribution_study/bs_to_mumu/bs_observables.png",
               lumi=_LUMIS),


rule reconstructed_distribution_ks_to_pipi:
    input:
        dist=f"{_OUTDIR}/{{lumi}}/reconstruction/ks_to_pipi/dist.parquet",
        summary=f"{_OUTDIR}/{{lumi}}/reconstruction/ks_to_pipi/dist_summary.json",
        script="physics/analysis/reconstructed_distribution_study/ks_to_pipi.py",
    output:
        f"{_OUTDIR}/{{lumi}}/reconstructed_distribution_study/ks_to_pipi/ks_observables.png",
    log:
        f"{_OUTDIR}/{{lumi}}/reconstructed_distribution_study/ks_to_pipi/run.log",
    params:
        outdir=_OUTDIR,
    shell:
        "PYTHONPATH=src python3 physics/analysis/reconstructed_distribution_study/ks_to_pipi.py"
        " --dist {input.dist} --summary {input.summary}"
        " --out-dir {params.outdir}/{wildcards.lumi}/reconstructed_distribution_study/ks_to_pipi"
        " --lumi {wildcards.lumi}"
        " > {log} 2>&1"


rule reconstructed_distribution_bs_to_mumu:
    input:
        dist=f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/dist.parquet",
        summary=f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/dist_summary.json",
        script="physics/analysis/reconstructed_distribution_study/bs_to_mumu.py",
    output:
        f"{_OUTDIR}/{{lumi}}/reconstructed_distribution_study/bs_to_mumu/bs_observables.png",
    log:
        f"{_OUTDIR}/{{lumi}}/reconstructed_distribution_study/bs_to_mumu/run.log",
    params:
        outdir=_OUTDIR,
    shell:
        "PYTHONPATH=src python3 physics/analysis/reconstructed_distribution_study/bs_to_mumu.py"
        " --dist {input.dist} --summary {input.summary}"
        " --out-dir {params.outdir}/{wildcards.lumi}/reconstructed_distribution_study/bs_to_mumu"
        " --lumi {wildcards.lumi}"
        " > {log} 2>&1"
