_OUTDIR = config["output_dir"]
_LUMIS = ["1p5e34"]
_SCRIPT = "physics/analysis/truth_distribution_study/signal_distributions.py"


rule all_truth_distribution_study:
    input:
        expand(
            f"{_OUTDIR}/{{lumi}}/truth_distribution_study/ks_to_pipi/ks_signal_distributions.png",
            lumi=_LUMIS,
        ),
        expand(
            f"{_OUTDIR}/{{lumi}}/truth_distribution_study/bs_to_mumu/bs_signal_distributions.png",
            lumi=_LUMIS,
        ),
        expand(
            f"{_OUTDIR}/{{lumi}}/truth_distribution_study/bs_to_jpsiphi/bs_signal_distributions.png",
            lumi=_LUMIS,
        ),


rule truth_distribution_ks_to_pipi:
    input:
        cheated=lambda wc: f"{config['output_dir']}/{wc.lumi}/reconstruction/ks_to_pipi/cheated_{config['luminosities'][wc.lumi]['channels']['ks_to_pipi']['modes']['full']['max_events']}.parquet",
        script=_SCRIPT,
    output:
        f"{_OUTDIR}/{{lumi}}/truth_distribution_study/ks_to_pipi/ks_signal_distributions.png",
    log:
        f"{_OUTDIR}/{{lumi}}/truth_distribution_study/ks_to_pipi/run.log",
    params:
        outdir=_OUTDIR,
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --cheated {input.cheated}"
        " --channel ks_to_pipi"
        " --out-dir {params.outdir}/{wildcards.lumi}/truth_distribution_study/ks_to_pipi"
        " --lumi {wildcards.lumi}"
        " > {log} 2>&1"


rule truth_distribution_bs_to_mumu:
    input:
        cheated=lambda wc: f"{config['output_dir']}/{wc.lumi}/reconstruction/bs_to_mumu/cheated_{config['luminosities'][wc.lumi]['channels']['bs_to_mumu']['modes']['full']['max_events']}.parquet",
        script=_SCRIPT,
    output:
        f"{_OUTDIR}/{{lumi}}/truth_distribution_study/bs_to_mumu/bs_signal_distributions.png",
    log:
        f"{_OUTDIR}/{{lumi}}/truth_distribution_study/bs_to_mumu/run.log",
    params:
        outdir=_OUTDIR,
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --cheated {input.cheated}"
        " --channel bs_to_mumu"
        " --out-dir {params.outdir}/{wildcards.lumi}/truth_distribution_study/bs_to_mumu"
        " --lumi {wildcards.lumi}"
        " > {log} 2>&1"


rule truth_distribution_bs_to_jpsiphi:
    input:
        cheated=lambda wc: f"{config['output_dir']}/{wc.lumi}/reconstruction/bs_to_jpsiphi/cheated_{config['luminosities'][wc.lumi]['channels']['bs_to_jpsiphi']['modes']['full']['max_events']}.parquet",
        script=_SCRIPT,
    output:
        f"{_OUTDIR}/{{lumi}}/truth_distribution_study/bs_to_jpsiphi/bs_signal_distributions.png",
    log:
        f"{_OUTDIR}/{{lumi}}/truth_distribution_study/bs_to_jpsiphi/run.log",
    params:
        outdir=_OUTDIR,
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --cheated {input.cheated}"
        " --channel bs_to_jpsiphi"
        " --out-dir {params.outdir}/{wildcards.lumi}/truth_distribution_study/bs_to_jpsiphi"
        " --lumi {wildcards.lumi}"
        " > {log} 2>&1"
