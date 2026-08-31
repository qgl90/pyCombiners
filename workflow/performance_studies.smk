# Long-track PID and tracking efficiency/fake-rate studies.


configfile: "config/performance_studies.yaml"


include: "common.smk"


_RECO = outdir("reconstruction")
_PID = outdir("analysis", "pid")
_TRACKING = outdir("analysis", "tracking")
_LABEL = config.get("label", "sample")
_PID_TAG = f"pid_{_LABEL}"
_TRACK_TYPES = ["long", "down", "longft", "longmp"]


rule all:
    input:
        f"{_RECO}/pid_tracks.parquet",
        f"{_RECO}/tracking_particles.parquet",
        f"{_PID}/{_PID_TAG}_roc_global.png",
        f"{_PID}/{_PID_TAG}_roc_vs_eta.png",
        f"{_PID}/{_PID_TAG}_roc_vs_p.png",
        f"{_PID}/{_PID_TAG}_roc_vs_pt.png",
        f"{_PID}/{_PID_TAG}_eff_vs_eta_fixed_misid.png",
        f"{_PID}/{_PID_TAG}_eff_vs_p_fixed_misid.png",
        f"{_PID}/{_PID_TAG}_eff_vs_pt_fixed_misid.png",
        f"{_PID}/{_PID_TAG}_maps_2d.png",
        expand(
            f"{_TRACKING}/{{track_type}}/tracking_performance_binned_{_LABEL}.parquet",
            track_type=_TRACK_TYPES,
        ),
        expand(
            f"{_TRACKING}/{{track_type}}/tracking_efficiency_{_LABEL}.png",
            track_type=_TRACK_TYPES,
        ),
        expand(
            f"{_TRACKING}/{{track_type}}/tracking_ghost_rate_{_LABEL}.png",
            track_type=_TRACK_TYPES,
        ),


rule pid_performance:
    input:
        script="physics/reconstruction/pid_performance.py",
    output:
        parquet=f"{_RECO}/pid_tracks.parquet",
        roc=f"{_PID}/{_PID_TAG}_roc_global.png",
        roc_eta=f"{_PID}/{_PID_TAG}_roc_vs_eta.png",
        roc_p=f"{_PID}/{_PID_TAG}_roc_vs_p.png",
        roc_pt=f"{_PID}/{_PID_TAG}_roc_vs_pt.png",
        eff_eta=f"{_PID}/{_PID_TAG}_eff_vs_eta_fixed_misid.png",
        eff_p=f"{_PID}/{_PID_TAG}_eff_vs_p_fixed_misid.png",
        eff_pt=f"{_PID}/{_PID_TAG}_eff_vs_pt_fixed_misid.png",
        maps=f"{_PID}/{_PID_TAG}_maps_2d.png",
    log:
        f"{_RECO}/pid_performance.log",
    params:
        data=config["input"],
        max_events=config["max_events"],
        outdir=_PID,
        tag=_PID_TAG,
    threads: workflow.cores
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --input '{params.data}'"
        " --max-events {params.max_events}"
        " --workers {threads}"
        " --out-dir {params.outdir}"
        " --out-tag {params.tag}"
        " --out-file {output.parquet}"
        " > {log} 2>&1"


rule tracking_performance:
    input:
        script="physics/reconstruction/tracking_efficiency.py",
    output:
        parquet=f"{_RECO}/tracking_particles.parquet",
        binned=expand(
            f"{_TRACKING}/{{track_type}}/tracking_performance_binned_{_LABEL}.parquet",
            track_type=_TRACK_TYPES,
        ),
        efficiency=expand(
            f"{_TRACKING}/{{track_type}}/tracking_efficiency_{_LABEL}.png",
            track_type=_TRACK_TYPES,
        ),
        ghost=expand(
            f"{_TRACKING}/{{track_type}}/tracking_ghost_rate_{_LABEL}.png",
            track_type=_TRACK_TYPES,
        ),
    log:
        f"{_RECO}/tracking_efficiency.log",
    params:
        data=config["input"],
        max_events=config["max_events"],
        outdir=_TRACKING,
        label=_LABEL,
    threads: workflow.cores
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --input '{params.data}'"
        " --max-events {params.max_events}"
        " --workers {threads}"
        " --out {output.parquet}"
        " --plot-dir {params.outdir}"
        " --label {params.label}"
        " --all-track-types"
        " > {log} 2>&1"
