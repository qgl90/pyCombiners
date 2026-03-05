_OUTDIR = config["output_dir"]
_LUMIS = ["1p5e34", "1p3e34", "1p0e34", "run3"]


rule all_PV_timing_association_study:
    input:
        expand(f"{_OUTDIR}/{{lumi}}/PV_timing_association_study/track_pv/track_pv_association.png",
               lumi=_LUMIS),
        expand(f"{_OUTDIR}/{{lumi}}/PV_timing_association_study/track_pv/pv_assoc_efficiency.png",
               lumi=_LUMIS),


rule pv_timing_association_track_pv:
    input:
        stats=f"{_OUTDIR}/{{lumi}}/reconstruction/track_pv_association/pv_stats.parquet",
        assoc=f"{_OUTDIR}/{{lumi}}/reconstruction/track_pv_association/pv_assoc.parquet",
        summary=f"{_OUTDIR}/{{lumi}}/reconstruction/track_pv_association/summary.json",
        script="physics/analysis/PV_timing_association_study/track_pv.py",
    output:
        f"{_OUTDIR}/{{lumi}}/PV_timing_association_study/track_pv/track_pv_association.png",
        f"{_OUTDIR}/{{lumi}}/PV_timing_association_study/track_pv/pv_assoc_efficiency.png",
    log:
        f"{_OUTDIR}/{{lumi}}/PV_timing_association_study/track_pv/run.log",
    params:
        outdir=_OUTDIR,
    shell:
        "PYTHONPATH=src python3 physics/analysis/PV_timing_association_study/track_pv.py"
        " --input-dir {params.outdir}/{wildcards.lumi}/reconstruction/track_pv_association"
        " --out-dir {params.outdir}/{wildcards.lumi}/PV_timing_association_study/track_pv"
        " --lumi {wildcards.lumi}"
        " > {log} 2>&1"
