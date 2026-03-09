_OUTDIR = config["output_dir"]
_LUMIS = ["1p5e34"]


rule all_pv_association_study:
    input:
        expand(
            f"{_OUTDIR}/{{lumi}}/pv_association_study/bs_to_mumu/pv_association_correctness.png",
            lumi=_LUMIS,
        ),
        expand(
            f"{_OUTDIR}/{{lumi}}/pv_association_study/track_pv/track_pv_timing_scan.png",
            lumi=_LUMIS,
        ),


rule pv_association_bs_to_mumu:
    input:
        full=f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/full.parquet",
        full_notime=f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/full_notime.parquet",
        full_pvtag=f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/full_pvtag.parquet",
        script="physics/analysis/pv_association_study/pv_association_correctness.py",
    output:
        f"{_OUTDIR}/{{lumi}}/pv_association_study/bs_to_mumu/pv_association_correctness.png",
    log:
        f"{_OUTDIR}/{{lumi}}/pv_association_study/bs_to_mumu/run.log",
    params:
        outdir=_OUTDIR,
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --data-dir {params.outdir}/{wildcards.lumi}/reconstruction/bs_to_mumu"
        " --out-dir {params.outdir}/{wildcards.lumi}/pv_association_study/bs_to_mumu"
        " --lumi {wildcards.lumi}"
        " > {log} 2>&1"


rule pv_association_track_pv_timing_scan:
    input:
        assoc=f"{_OUTDIR}/{{lumi}}/reconstruction/track_pv_association/pv_assoc.parquet",
        script="physics/analysis/pv_association_study/track_pv_timing_scan.py",
    output:
        f"{_OUTDIR}/{{lumi}}/pv_association_study/track_pv/track_pv_timing_scan.png",
    log:
        f"{_OUTDIR}/{{lumi}}/pv_association_study/track_pv/run.log",
    params:
        outdir=_OUTDIR,
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --input-dir {params.outdir}/{wildcards.lumi}/reconstruction/track_pv_association"
        " --out-dir {params.outdir}/{wildcards.lumi}/pv_association_study/track_pv"
        " --lumi {wildcards.lumi}"
        " > {log} 2>&1"
