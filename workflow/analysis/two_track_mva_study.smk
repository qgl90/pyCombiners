_OUTDIR = config["output_dir"]
_LUMIS = ["1p5e34"]


rule all_two_track_mva_study:
    input:
        expand(f"{_OUTDIR}/{{lumi}}/two_track_mva_study/mva_response.png",
               lumi=_LUMIS),
        expand(f"{_OUTDIR}/{{lumi}}/two_track_mva_study/mass.png",
               lumi=_LUMIS),
        expand(f"{_OUTDIR}/{{lumi}}/two_track_mva_study/observables.png",
               lumi=_LUMIS),


rule two_track_mva_study:
    input:
        data=f"{_OUTDIR}/{{lumi}}/reconstruction/two_track_mva/mva.parquet",
        script="physics/analysis/two_track_mva_study/two_track_mva.py",
    output:
        f"{_OUTDIR}/{{lumi}}/two_track_mva_study/mva_response.png",
        f"{_OUTDIR}/{{lumi}}/two_track_mva_study/mass.png",
        f"{_OUTDIR}/{{lumi}}/two_track_mva_study/observables.png",
    log:
        f"{_OUTDIR}/{{lumi}}/two_track_mva_study/run.log",
    params:
        outdir=_OUTDIR,
    shell:
        "PYTHONPATH=src python3 physics/analysis/two_track_mva_study/two_track_mva.py"
        " --input {input.data}"
        " --out-dir {params.outdir}/{wildcards.lumi}/two_track_mva_study"
        " --lumi {wildcards.lumi}"
        " > {log} 2>&1"
