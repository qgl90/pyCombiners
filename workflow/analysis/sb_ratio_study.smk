_OUTDIR = config["output_dir"]
_LUMIS = list(config["luminosities"].keys())


rule all_sb_ratio_study:
    input:
        expand(f"{_OUTDIR}/{{lumi}}/sb_ratio_study/bs_to_mumu/bs_sb_vs_eta.png",
               lumi=["1p5e34"]),
        expand(f"{_OUTDIR}/{{lumi}}/sb_ratio_study/bs_to_mumu/bs_mass_vs_eta.png",
               lumi=["1p5e34"]),
        f"{_OUTDIR}/sb_ratio_study/bs_to_mumu_vs_lumi/bs_sb_vs_lumi.png",
        f"{_OUTDIR}/sb_ratio_study/bs_to_mumu_vs_lumi/bs_sb_vs_eta_by_lumi.png",
        f"{_OUTDIR}/sb_ratio_study/bs_to_mumu_vs_lumi/bs_mass_by_lumi.png",


rule sb_ratio_bs_to_mumu:
    input:
        full=f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/full.parquet",
        script="physics/analysis/sb_ratio_study/bs_to_mumu.py",
    output:
        f"{_OUTDIR}/{{lumi}}/sb_ratio_study/bs_to_mumu/bs_sb_vs_eta.png",
        f"{_OUTDIR}/{{lumi}}/sb_ratio_study/bs_to_mumu/bs_mass_vs_eta.png",
    log:
        f"{_OUTDIR}/{{lumi}}/sb_ratio_study/bs_to_mumu/run.log",
    params:
        outdir=_OUTDIR,
    shell:
        "PYTHONPATH=src python3 physics/analysis/sb_ratio_study/bs_to_mumu.py"
        " --full {input.full}"
        " --out-dir {params.outdir}/{wildcards.lumi}/sb_ratio_study/bs_to_mumu"
        " --lumi {wildcards.lumi}"
        " > {log} 2>&1"


rule sb_ratio_bs_to_mumu_vs_lumi:
    input:
        expand(f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/full.parquet",
               lumi=_LUMIS),
        script="physics/analysis/sb_ratio_study/bs_to_mumu_vs_lumi.py",
    output:
        f"{_OUTDIR}/sb_ratio_study/bs_to_mumu_vs_lumi/bs_sb_vs_lumi.png",
        f"{_OUTDIR}/sb_ratio_study/bs_to_mumu_vs_lumi/bs_sb_vs_eta_by_lumi.png",
        f"{_OUTDIR}/sb_ratio_study/bs_to_mumu_vs_lumi/bs_mass_by_lumi.png",
    log:
        f"{_OUTDIR}/sb_ratio_study/bs_to_mumu_vs_lumi/run.log",
    params:
        outdir=_OUTDIR,
        input_args=" ".join(
            f"{lumi}={_OUTDIR}/{lumi}/reconstruction/bs_to_mumu/full.parquet"
            for lumi in _LUMIS
        ),
    shell:
        "PYTHONPATH=src python3 physics/analysis/sb_ratio_study/bs_to_mumu_vs_lumi.py"
        " --inputs {params.input_args}"
        " --out-dir {params.outdir}/sb_ratio_study/bs_to_mumu_vs_lumi"
        " > {log} 2>&1"
