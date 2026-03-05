_OUTDIR = config["output_dir"]
_LUMIS = ["1p5e34"]


rule all_decay_performance_study:
    input:
        expand(f"{_OUTDIR}/{{lumi}}/decay_performance_study/ks_to_pipi/ks_eff_vs_kinematics.png",
               lumi=_LUMIS),
        expand(f"{_OUTDIR}/{{lumi}}/decay_performance_study/ks_to_pipi/ks_mass.png",
               lumi=_LUMIS),
        expand(f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_mumu/bs_eff_vs_kinematics.png",
               lumi=_LUMIS),
        expand(f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_mumu/bs_mass.png",
               lumi=_LUMIS),
        expand(f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_mumu_notime/bs_eff_vs_kinematics.png",
               lumi=_LUMIS),
        expand(f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_mumu_notime/bs_mass.png",
               lumi=_LUMIS),
        expand(f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_mumu_pvtag/bs_eff_vs_kinematics.png",
               lumi=_LUMIS),
        expand(f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_mumu_pvtag/bs_mass.png",
               lumi=_LUMIS),


rule decay_performance_ks_to_pipi:
    input:
        cheated=f"{_OUTDIR}/{{lumi}}/reconstruction/ks_to_pipi/full_cheated.parquet",
        full=f"{_OUTDIR}/{{lumi}}/reconstruction/ks_to_pipi/full.parquet",
        script="physics/analysis/decay_performance_study/ks_to_pipi.py",
    output:
        f"{_OUTDIR}/{{lumi}}/decay_performance_study/ks_to_pipi/ks_eff_vs_kinematics.png",
        f"{_OUTDIR}/{{lumi}}/decay_performance_study/ks_to_pipi/ks_mass.png",
    log:
        f"{_OUTDIR}/{{lumi}}/decay_performance_study/ks_to_pipi/run.log",
    params:
        outdir=_OUTDIR,
    shell:
        "PYTHONPATH=src python3 physics/analysis/decay_performance_study/ks_to_pipi.py"
        " --cheated {input.cheated} --full {input.full}"
        " --out-dir {params.outdir}/{wildcards.lumi}/decay_performance_study/ks_to_pipi"
        " --lumi {wildcards.lumi}"
        " > {log} 2>&1"


rule decay_performance_bs_to_mumu:
    input:
        cheated=f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/full_cheated.parquet",
        full=f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/full.parquet",
        script="physics/analysis/decay_performance_study/bs_to_mumu.py",
    output:
        f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_mumu/bs_eff_vs_kinematics.png",
        f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_mumu/bs_mass.png",
    log:
        f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_mumu/run.log",
    params:
        outdir=_OUTDIR,
    shell:
        "PYTHONPATH=src python3 physics/analysis/decay_performance_study/bs_to_mumu.py"
        " --cheated {input.cheated} --full {input.full}"
        " --out-dir {params.outdir}/{wildcards.lumi}/decay_performance_study/bs_to_mumu"
        " --lumi {wildcards.lumi}"
        " > {log} 2>&1"


rule decay_performance_bs_to_mumu_notime:
    input:
        cheated=f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/full_notime_cheated.parquet",
        full_notime=f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/full_notime.parquet",
        script="physics/analysis/decay_performance_study/bs_to_mumu_notime.py",
    output:
        f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_mumu_notime/bs_eff_vs_kinematics.png",
        f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_mumu_notime/bs_mass.png",
    log:
        f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_mumu_notime/run.log",
    params:
        outdir=_OUTDIR,
    shell:
        "PYTHONPATH=src python3 physics/analysis/decay_performance_study/bs_to_mumu_notime.py"
        " --cheated {input.cheated} --full {input.full_notime}"
        " --out-dir {params.outdir}/{wildcards.lumi}/decay_performance_study/bs_to_mumu_notime"
        " --lumi {wildcards.lumi}"
        " > {log} 2>&1"


rule decay_performance_bs_to_mumu_pvtag:
    input:
        cheated=f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/pvtag_cheated.parquet",
        full=f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/pvtag.parquet",
        script="physics/analysis/decay_performance_study/bs_to_mumu.py",
    output:
        f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_mumu_pvtag/bs_eff_vs_kinematics.png",
        f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_mumu_pvtag/bs_mass.png",
    log:
        f"{_OUTDIR}/{{lumi}}/decay_performance_study/bs_to_mumu_pvtag/run.log",
    params:
        outdir=_OUTDIR,
    shell:
        "PYTHONPATH=src python3 physics/analysis/decay_performance_study/bs_to_mumu.py"
        " --cheated {input.cheated} --full {input.full}"
        " --out-dir {params.outdir}/{wildcards.lumi}/decay_performance_study/bs_to_mumu_pvtag"
        " --lumi {wildcards.lumi}"
        " > {log} 2>&1"
