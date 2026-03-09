_OUTDIR = config["output_dir"]
_LUMIS = ["1p5e34"]
_SCRIPT = "physics/analysis/decay_performance_study/decay_performance.py"


rule all_decay_performance_study:
    input:
        expand(
            f"{_OUTDIR}/{{lumi}}/decay_performance_study/ks_to_pipi/ks_eff_vs_kinematics.png",
            lumi=_LUMIS,
        ),
        expand(
            f"{_OUTDIR}/{{lumi}}/decay_performance_study/ks_to_pipi/ks_mass.png",
            lumi=_LUMIS,
        ),
        expand(
            f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_mumu/bs_eff_vs_kinematics.png",
            lumi=_LUMIS,
        ),
        expand(
            f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_mumu/bs_mass.png",
            lumi=_LUMIS,
        ),
        expand(
            f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_mumu_notime/bs_eff_vs_kinematics.png",
            lumi=_LUMIS,
        ),
        expand(
            f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_mumu_notime/bs_mass.png",
            lumi=_LUMIS,
        ),
        expand(
            f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_mumu_pvtag/bs_eff_vs_kinematics.png",
            lumi=_LUMIS,
        ),
        expand(
            f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_mumu_pvtag/bs_mass.png",
            lumi=_LUMIS,
        ),
        expand(
            f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_jpsiphi/bs_eff_vs_kinematics.png",
            lumi=_LUMIS,
        ),
        expand(
            f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_jpsiphi/bs_mass.png",
            lumi=_LUMIS,
        ),
        expand(
            f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_jpsiphi/bs_jpsi_mass.png",
            lumi=_LUMIS,
        ),
        expand(
            f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_jpsiphi/bs_phi_mass.png",
            lumi=_LUMIS,
        ),


rule decay_performance_ks_to_pipi:
    input:
        cheated=lambda wc: f"{config['output_dir']}/{wc.lumi}/reconstruction/ks_to_pipi/cheated_{config['luminosities'][wc.lumi]['channels']['ks_to_pipi']['modes']['full']['max_events']}.parquet",
        full=f"{_OUTDIR}/{{lumi}}/reconstruction/ks_to_pipi/full.parquet",
        script=_SCRIPT,
    output:
        f"{_OUTDIR}/{{lumi}}/decay_performance_study/ks_to_pipi/ks_eff_vs_kinematics.png",
        f"{_OUTDIR}/{{lumi}}/decay_performance_study/ks_to_pipi/ks_mass.png",
    log:
        f"{_OUTDIR}/{{lumi}}/decay_performance_study/ks_to_pipi/run.log",
    params:
        outdir=_OUTDIR,
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --cheated {input.cheated} --full {input.full}"
        " --channel ks_to_pipi"
        " --out-dir {params.outdir}/{wildcards.lumi}/decay_performance_study/ks_to_pipi"
        " --lumi {wildcards.lumi}"
        " > {log} 2>&1"


rule decay_performance_bs_to_mumu:
    input:
        cheated=lambda wc: f"{config['output_dir']}/{wc.lumi}/reconstruction/bs_to_mumu/cheated_{config['luminosities'][wc.lumi]['channels']['bs_to_mumu']['modes']['full']['max_events']}.parquet",
        full=f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/full.parquet",
        script=_SCRIPT,
    output:
        f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_mumu/bs_eff_vs_kinematics.png",
        f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_mumu/bs_mass.png",
    log:
        f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_mumu/run.log",
    params:
        outdir=_OUTDIR,
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --cheated {input.cheated} --full {input.full}"
        " --channel bs_to_mumu"
        " --out-dir {params.outdir}/{wildcards.lumi}/decay_performance_study/bs_to_mumu"
        " --lumi {wildcards.lumi}"
        " > {log} 2>&1"


rule decay_performance_bs_to_mumu_notime:
    input:
        cheated=lambda wc: f"{config['output_dir']}/{wc.lumi}/reconstruction/bs_to_mumu/cheated_{config['luminosities'][wc.lumi]['channels']['bs_to_mumu']['modes']['full_notime']['max_events']}.parquet",
        full=f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/full_notime.parquet",
        script=_SCRIPT,
    output:
        f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_mumu_notime/bs_eff_vs_kinematics.png",
        f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_mumu_notime/bs_mass.png",
    log:
        f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_mumu_notime/run.log",
    params:
        outdir=_OUTDIR,
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --cheated {input.cheated} --full {input.full}"
        " --channel bs_to_mumu"
        " --tag 'NO TIMING'"
        " --out-dir {params.outdir}/{wildcards.lumi}/decay_performance_study/bs_to_mumu_notime"
        " --lumi {wildcards.lumi}"
        " > {log} 2>&1"


rule decay_performance_bs_to_mumu_pvtag:
    input:
        cheated=lambda wc: f"{config['output_dir']}/{wc.lumi}/reconstruction/bs_to_mumu/cheated_{config['luminosities'][wc.lumi]['channels']['bs_to_mumu_pvtag']['max_events']}.parquet",
        full=f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/full_pvtag.parquet",
        script=_SCRIPT,
    output:
        f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_mumu_pvtag/bs_eff_vs_kinematics.png",
        f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_mumu_pvtag/bs_mass.png",
    log:
        f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_mumu_pvtag/run.log",
    params:
        outdir=_OUTDIR,
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --cheated {input.cheated} --full {input.full}"
        " --channel bs_to_mumu"
        " --out-dir {params.outdir}/{wildcards.lumi}/decay_performance_study/bs_to_mumu_pvtag"
        " --lumi {wildcards.lumi}"
        " > {log} 2>&1"


rule decay_performance_bs_to_jpsiphi:
    input:
        cheated=lambda wc: f"{config['output_dir']}/{wc.lumi}/reconstruction/bs_to_jpsiphi/cheated_{config['luminosities'][wc.lumi]['channels']['bs_to_jpsiphi']['modes']['full']['max_events']}.parquet",
        full=f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_jpsiphi/full.parquet",
        script=_SCRIPT,
    output:
        f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_jpsiphi/bs_eff_vs_kinematics.png",
        f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_jpsiphi/bs_mass.png",
        f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_jpsiphi/bs_jpsi_mass.png",
        f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_jpsiphi/bs_phi_mass.png",
    log:
        f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_jpsiphi/run.log",
    params:
        outdir=_OUTDIR,
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --cheated {input.cheated} --full {input.full}"
        " --channel bs_to_jpsiphi"
        " --out-dir {params.outdir}/{wildcards.lumi}/decay_performance_study/bs_to_jpsiphi"
        " --lumi {wildcards.lumi}"
        " > {log} 2>&1"
