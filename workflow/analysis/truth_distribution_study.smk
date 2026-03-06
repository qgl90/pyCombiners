_OUTDIR = config["output_dir"]
_LUMIS = ["1p5e34"]


rule all_truth_distribution_study:
    input:
        expand(f"{_OUTDIR}/{{lumi}}/truth_distribution_study/ks_to_pipi/ks_signal_distributions.png",
               lumi=_LUMIS),
        expand(f"{_OUTDIR}/{{lumi}}/truth_distribution_study/bs_to_mumu/bs_signal_distributions.png",
               lumi=_LUMIS),


rule truth_distribution_ks_to_pipi:
    input:
        cheated=f"{_OUTDIR}/{{lumi}}/reconstruction/ks_to_pipi/cheated.parquet",
        summary=f"{_OUTDIR}/{{lumi}}/reconstruction/ks_to_pipi/cheated_summary.json",
        script="physics/analysis/truth_distribution_study/ks_to_pipi.py",
    output:
        f"{_OUTDIR}/{{lumi}}/truth_distribution_study/ks_to_pipi/ks_signal_distributions.png",
    log:
        f"{_OUTDIR}/{{lumi}}/truth_distribution_study/ks_to_pipi/run.log",
    params:
        outdir=_OUTDIR,
    shell:
        "PYTHONPATH=src python3 physics/analysis/truth_distribution_study/ks_to_pipi.py"
        " --cheated {input.cheated} --summary {input.summary}"
        " --out-dir {params.outdir}/{wildcards.lumi}/truth_distribution_study/ks_to_pipi"
        " --lumi {wildcards.lumi}"
        " > {log} 2>&1"


rule truth_distribution_bs_to_mumu:
    input:
        cheated=f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/cheated.parquet",
        summary=f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/cheated_summary.json",
        script="physics/analysis/truth_distribution_study/bs_to_mumu.py",
    output:
        f"{_OUTDIR}/{{lumi}}/truth_distribution_study/bs_to_mumu/bs_signal_distributions.png",
    log:
        f"{_OUTDIR}/{{lumi}}/truth_distribution_study/bs_to_mumu/run.log",
    params:
        outdir=_OUTDIR,
    shell:
        "PYTHONPATH=src python3 physics/analysis/truth_distribution_study/bs_to_mumu.py"
        " --cheated {input.cheated} --summary {input.summary}"
        " --out-dir {params.outdir}/{wildcards.lumi}/truth_distribution_study/bs_to_mumu"
        " --lumi {wildcards.lumi}"
        " > {log} 2>&1"
