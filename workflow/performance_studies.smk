# Long-track PID and tracking efficiency/fake-rate studies.


configfile: "config/performance_studies.yaml"


include: "common.smk"


_RECO = outdir("reconstruction")
_PID = outdir("analysis", "pid")
_TRACKING = outdir("analysis", "tracking")
_PV = outdir("analysis", "pv")
_LABEL = config.get("label", "sample")
_PID_TAG = f"pid_{_LABEL}"
_TRACK_TYPES = ["long", "down", "longft", "longmp"]
_PV_COORDINATES = ["x", "y", "z", "time"]


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
        expand(
            f"{_TRACKING}/{{track_type}}/tracking_chi2ndof_scan_binned_{_LABEL}.parquet",
            track_type=_TRACK_TYPES,
        ),
        expand(
            f"{_TRACKING}/{{track_type}}/tracking_ghost_rate_chi2ndof_scan_{_LABEL}.png",
            track_type=_TRACK_TYPES,
        ),
        expand(
            f"{_TRACKING}/{{track_type}}/tracking_performance_chi2ndof_scan_{_LABEL}.png",
            track_type=_TRACK_TYPES,
        ),
        f"{_TRACKING}/momentum_resolution/momentum_resolution_binned_{_LABEL}.parquet",
        f"{_TRACKING}/momentum_resolution/deltap_over_p_vs_p_{_LABEL}.png",
        f"{_TRACKING}/momentum_resolution/deltap_over_p_vs_eta_{_LABEL}.png",
        f"{_TRACKING}/momentum_resolution/deltap_over_p_vs_phi_{_LABEL}.png",
        f"{_TRACKING}/momentum_resolution/gaussian_fit_checks_vs_p_{_LABEL}.pdf",
        f"{_TRACKING}/momentum_resolution/gaussian_fit_checks_vs_eta_{_LABEL}.pdf",
        f"{_TRACKING}/momentum_resolution/gaussian_fit_checks_vs_phi_{_LABEL}.pdf",
        f"{_PV}/pv_resolution_fits_{_LABEL}.parquet",
        f"{_PV}/pv_pull_fits_{_LABEL}.parquet",
        f"{_PV}/pv_pull_global_fits_{_LABEL}.parquet",
        f"{_PV}/pv_residuals_vs_ndof_{_LABEL}.png",
        f"{_PV}/pv_bias_resolution_vs_ndof_{_LABEL}.png",
        f"{_PV}/pv_pulls_{_LABEL}.png",
        f"{_PV}/pv_pull_mean_width_vs_ndof_{_LABEL}.png",
        expand(
            f"{_PV}/pv_gaussian_fit_checks_{{coordinate}}_{_LABEL}.png",
            coordinate=_PV_COORDINATES,
        ),
        expand(
            f"{_PV}/pv_pull_gaussian_fit_checks_{{coordinate}}_{_LABEL}.png",
            coordinate=_PV_COORDINATES,
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
        chi2_scan_table=expand(
            f"{_TRACKING}/{{track_type}}/tracking_chi2ndof_scan_binned_{_LABEL}.parquet",
            track_type=_TRACK_TYPES,
        ),
        chi2_scan_ghost=expand(
            f"{_TRACKING}/{{track_type}}/tracking_ghost_rate_chi2ndof_scan_{_LABEL}.png",
            track_type=_TRACK_TYPES,
        ),
        chi2_scan_summary=expand(
            f"{_TRACKING}/{{track_type}}/tracking_performance_chi2ndof_scan_{_LABEL}.png",
            track_type=_TRACK_TYPES,
        ),
        resolution_table=(
            f"{_TRACKING}/momentum_resolution/"
            f"momentum_resolution_binned_{_LABEL}.parquet"
        ),
        resolution_p=(
            f"{_TRACKING}/momentum_resolution/"
            f"deltap_over_p_vs_p_{_LABEL}.png"
        ),
        resolution_eta=(
            f"{_TRACKING}/momentum_resolution/"
            f"deltap_over_p_vs_eta_{_LABEL}.png"
        ),
        resolution_phi=(
            f"{_TRACKING}/momentum_resolution/"
            f"deltap_over_p_vs_phi_{_LABEL}.png"
        ),
        fit_checks_p=(
            f"{_TRACKING}/momentum_resolution/"
            f"gaussian_fit_checks_vs_p_{_LABEL}.pdf"
        ),
        fit_checks_eta=(
            f"{_TRACKING}/momentum_resolution/"
            f"gaussian_fit_checks_vs_eta_{_LABEL}.pdf"
        ),
        fit_checks_phi=(
            f"{_TRACKING}/momentum_resolution/"
            f"gaussian_fit_checks_vs_phi_{_LABEL}.pdf"
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


rule pv_resolution:
    input:
        script="src/pv/pv_resolution.py",
    output:
        parquet=f"{_RECO}/pv_residuals.parquet",
        resolution_table=f"{_PV}/pv_resolution_fits_{_LABEL}.parquet",
        pull_table=f"{_PV}/pv_pull_fits_{_LABEL}.parquet",
        global_pull_table=f"{_PV}/pv_pull_global_fits_{_LABEL}.parquet",
        residuals=f"{_PV}/pv_residuals_vs_ndof_{_LABEL}.png",
        resolution=f"{_PV}/pv_bias_resolution_vs_ndof_{_LABEL}.png",
        pulls=f"{_PV}/pv_pulls_{_LABEL}.png",
        pull_calibration=f"{_PV}/pv_pull_mean_width_vs_ndof_{_LABEL}.png",
        residual_checks=expand(
            f"{_PV}/pv_gaussian_fit_checks_{{coordinate}}_{_LABEL}.png",
            coordinate=_PV_COORDINATES,
        ),
        pull_checks=expand(
            f"{_PV}/pv_pull_gaussian_fit_checks_{{coordinate}}_{_LABEL}.png",
            coordinate=_PV_COORDINATES,
        ),
    log:
        f"{_RECO}/pv_resolution.log",
    params:
        data=config["input"],
        max_events=config["max_events"],
        outdir=_PV,
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
        " > {log} 2>&1"
