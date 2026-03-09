_OUTDIR = "public/toy_study/vertex_fit_4d_vs_3d"
_SCRIPT = "physics/toy_study/vertex_fit_4d_vs_3d.py"


rule all_vertex_fit_4d_vs_3d:
    input:
        f"{_OUTDIR}/z_resolution_comparison.png",
        f"{_OUTDIR}/improvement_vs_speed_diff.png",
        f"{_OUTDIR}/pull_distributions.png",


rule vertex_fit_4d_vs_3d:
    input:
        script=_SCRIPT,
    output:
        f"{_OUTDIR}/z_resolution_comparison.png",
        f"{_OUTDIR}/improvement_vs_speed_diff.png",
        f"{_OUTDIR}/pull_distributions.png",
    log:
        f"{_OUTDIR}/run.log",
    params:
        outdir=_OUTDIR,
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --n-toys 10000"
        " --out-dir {params.outdir}"
        " > {log} 2>&1"
