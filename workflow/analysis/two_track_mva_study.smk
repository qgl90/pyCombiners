_OUTDIR = config["output_dir"]
_LUMIS = ["1p5e34"]
_SCRIPT = "physics/analysis/two_track_mva_study/mva_performance.py"


rule all_two_track_mva_study:
    input:
        expand(
            f"{_OUTDIR}/{{lumi}}/two_track_mva_study/bs_to_mumu/mva_response_distribution.png",
            lumi=_LUMIS,
        ),
        expand(
            f"{_OUTDIR}/{{lumi}}/two_track_mva_study/bs_to_mumu/tagged_pvs_per_event.png",
            lumi=_LUMIS,
        ),


rule two_track_mva_study_bs_to_mumu:
    input:
        data=f"{_OUTDIR}/{{lumi}}/reconstruction/two_track_mva/mva.parquet",
        script=_SCRIPT,
    output:
        f"{_OUTDIR}/{{lumi}}/two_track_mva_study/bs_to_mumu/mva_response_distribution.png",
        f"{_OUTDIR}/{{lumi}}/two_track_mva_study/bs_to_mumu/tagged_pvs_per_event.png",
    log:
        f"{_OUTDIR}/{{lumi}}/two_track_mva_study/bs_to_mumu/run.log",
    params:
        outdir=_OUTDIR,
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --input {input.data}"
        " --channel bs_to_mumu"
        " --out-dir {params.outdir}/{wildcards.lumi}/two_track_mva_study/bs_to_mumu"
        " --lumi {wildcards.lumi}"
        " > {log} 2>&1"
