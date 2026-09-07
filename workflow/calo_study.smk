# PicoCal calo study: photon-cluster dump + energy/time plots.


configfile: "config/calo_study.yaml"


include: "common.smk"


_RECO = outdir("reconstruction")
_STUDY = outdir("analysis")
_MATCHES = f"{_RECO}/bgamma_matches.parquet"
_SCRIPTS = "physics/analysis/calo_study"


rule all:
    input:
        f"{_STUDY}/cluster_splitting_profile.png",
        f"{_STUDY}/cluster_splitting_distance.png",
        f"{_STUDY}/cluster_splitting_lead_fraction.png",
        f"{_STUDY}/gamma_energy_residual_2d_direct_bs_rel.png",
        f"{_STUDY}/gamma_energy_residual_2d_direct_bs_rel_pt1p5.png",
        f"{_STUDY}/gamma_residual_bins_direct_bs.png",
        f"{_STUDY}/gamma_residual_bins_summary_direct_bs.png",
        f"{_STUDY}/gamma_residual_bins_direct_bs_pt1p5.png",
        f"{_STUDY}/gamma_residual_bins_summary_direct_bs_pt1p5.png",
        f"{_STUDY}/gamma_section_time_diff_xy.png",
        f"{_STUDY}/gamma_section_time_diff_map.png",
        f"{_STUDY}/gamma_section_time_diff_by_area.png",
        f"{_STUDY}/gamma_time_resolution_default.png",
        f"{_STUDY}/gamma_time_resolution_default_by_area.png",
        f"{_STUDY}/gamma_time_aligned_vs_et.png",
        f"{_STUDY}/gamma_time_aligned_vs_et_bins.png",


rule bgamma_matches:
    input:
        script="physics/reconstruction/calo_study.py",
    output:
        _MATCHES,
    log:
        f"{_RECO}/bgamma_matches.log",
    params:
        data=config["input"],
        max_events=config["max_events"],
    threads: workflow.cores
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --input '{params.data}'"
        " --max-events {params.max_events}"
        " --workers {threads}"
        " --out-file {output}"
        " > {log} 2>&1"


rule cluster_splitting:
    input:
        matches=_MATCHES,
        script=f"{_SCRIPTS}/cluster_splitting.py",
    output:
        f"{_STUDY}/cluster_splitting_profile.png",
        f"{_STUDY}/cluster_splitting_distance.png",
        f"{_STUDY}/cluster_splitting_lead_fraction.png",
    log:
        f"{_STUDY}/cluster_splitting.log",
    params:
        outdir=_STUDY,
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --input {input.matches}"
        " --out-dir {params.outdir}"
        " > {log} 2>&1"


rule energy_residual_2d:
    input:
        matches=_MATCHES,
        script=f"{_SCRIPTS}/gamma_energy_residual_2d.py",
    output:
        f"{_STUDY}/gamma_energy_residual_2d_direct_bs_rel.png",
    log:
        f"{_STUDY}/gamma_energy_residual_2d.log",
    params:
        outdir=_STUDY,
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --input {input.matches}"
        " --out-dir {params.outdir}"
        " > {log} 2>&1"


rule energy_residual_2d_pt:
    input:
        matches=_MATCHES,
        script=f"{_SCRIPTS}/gamma_energy_residual_2d.py",
    output:
        f"{_STUDY}/gamma_energy_residual_2d_direct_bs_rel_pt1p5.png",
    log:
        f"{_STUDY}/gamma_energy_residual_2d_pt1p5.log",
    params:
        outdir=_STUDY,
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --input {input.matches}"
        " --min-pt 1.5"
        " --out-dir {params.outdir}"
        " > {log} 2>&1"


rule residual_bins:
    input:
        matches=_MATCHES,
        script=f"{_SCRIPTS}/gamma_residual_bins.py",
    output:
        f"{_STUDY}/gamma_residual_bins_direct_bs.png",
        f"{_STUDY}/gamma_residual_bins_summary_direct_bs.png",
    log:
        f"{_STUDY}/gamma_residual_bins.log",
    params:
        outdir=_STUDY,
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --input {input.matches}"
        " --out-dir {params.outdir}"
        " > {log} 2>&1"


rule residual_bins_pt:
    input:
        matches=_MATCHES,
        script=f"{_SCRIPTS}/gamma_residual_bins.py",
    output:
        f"{_STUDY}/gamma_residual_bins_direct_bs_pt1p5.png",
        f"{_STUDY}/gamma_residual_bins_summary_direct_bs_pt1p5.png",
    log:
        f"{_STUDY}/gamma_residual_bins_pt1p5.log",
    params:
        outdir=_STUDY,
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --input {input.matches}"
        " --min-pt 1.5"
        " --out-dir {params.outdir}"
        " > {log} 2>&1"


rule section_time_diff:
    input:
        matches=_MATCHES,
        script=f"{_SCRIPTS}/section_time_diff.py",
    output:
        f"{_STUDY}/gamma_section_time_diff_xy.png",
        f"{_STUDY}/gamma_section_time_diff_map.png",
        f"{_STUDY}/gamma_section_time_diff_by_area.png",
    log:
        f"{_STUDY}/gamma_section_time_diff.log",
    params:
        outdir=_STUDY,
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --input {input.matches}"
        " --out-dir {params.outdir}"
        " > {log} 2>&1"


rule time_resolution:
    input:
        matches=_MATCHES,
        script=f"{_SCRIPTS}/gamma_time_resolution.py",
    output:
        f"{_STUDY}/gamma_time_resolution_default.png",
        f"{_STUDY}/gamma_time_resolution_default_by_area.png",
    log:
        f"{_STUDY}/gamma_time_resolution_default.log",
    params:
        outdir=_STUDY,
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --input {input.matches}"
        " --out-dir {params.outdir}"
        " > {log} 2>&1"


rule time_aligned:
    input:
        matches=_MATCHES,
        script=f"{_SCRIPTS}/gamma_time_aligned.py",
    output:
        f"{_STUDY}/gamma_time_aligned_vs_et.png",
        f"{_STUDY}/gamma_time_aligned_vs_et_bins.png",
    log:
        f"{_STUDY}/gamma_time_aligned.log",
    params:
        outdir=_STUDY,
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --input {input.matches}"
        " --out-dir {params.outdir}"
        " > {log} 2>&1"
