# Bs -> mu+ mu- pipeline: reconstruction (all modes), PV association,
# TwoTrackMVA and the S/B, performance and distribution plots.


configfile: "config/bs_to_mumu.yaml"


include: "common.smk"


_RECO = outdir("reconstruction")
_RECO_MODES = ["full", "full_notime", "dist"]


wildcard_constraints:
    bs_mode="full|full_notime|dist",
    n_events="[0-9]+",


def _cheated(n_events):
    return f"{_RECO}/cheated_{n_events}.parquet"


def _perf(variant):
    return outdir("analysis", f"decay_performance{variant}")


rule all:
    input:
        expand(f"{_RECO}/{{mode}}.parquet", mode=_RECO_MODES),
        f"{_RECO}/full_pvtag.parquet",
        f"{_RECO}/pv_assoc.parquet",
        f"{_RECO}/mva.parquet",
        [f"{_perf(v)}/bs_eff_vs_kinematics.png" for v in ("", "_notime", "_pvtag")],
        [f"{_perf(v)}/bs_mass.png" for v in ("", "_notime", "_pvtag")],
        outdir("analysis", "pv_association") + "/pv_association_correctness.png",
        outdir("analysis", "track_pv_timing") + "/track_pv_timing_scan.png",
        outdir("analysis", "sb_ratio") + "/sb_ratio_vs_eta.png",
        outdir("analysis", "sb_ratio") + "/mass_vs_eta.png",
        outdir("analysis", "truth_distributions") + "/bs_signal_distributions.png",
        outdir("analysis", "reconstructed_distributions") + "/bs_observables.png",
        outdir("analysis", "two_track_mva") + "/mva_response_distribution.png",


rule mumu_cheated:
    input:
        script="physics/reconstruction/bs_to_mumu.py",
    output:
        f"{_RECO}/cheated_{{n_events}}.parquet",
    log:
        f"{_RECO}/cheated_{{n_events}}.log",
    params:
        data=config["input"],
    threads: workflow.cores
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --mode cheated"
        " --input '{params.data}'"
        " --max-events {wildcards.n_events}"
        " --workers {threads}"
        " --out-file {output}"
        " > {log} 2>&1"


rule mumu_reco:
    input:
        cheated=lambda wc: _cheated(events(f"max_events_{wc.bs_mode}")),
        script="physics/reconstruction/bs_to_mumu.py",
    output:
        f"{_RECO}/{{bs_mode}}.parquet",
    log:
        f"{_RECO}/{{bs_mode}}.log",
    params:
        data=config["input"],
        max_events=lambda wc: events(f"max_events_{wc.bs_mode}"),
        outdir=_RECO,
    threads: workflow.cores
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --mode {wildcards.bs_mode}"
        " --input '{params.data}'"
        " --max-events {params.max_events}"
        " --workers {threads}"
        " --out-dir {params.outdir}"
        " > {log} 2>&1"


rule mumu_pvtag:
    input:
        model="models/two_track_mva.onnx",
        cheated=lambda wc: _cheated(events("max_events_pvtag")),
        script="physics/reconstruction/bs_to_mumu.py",
    output:
        f"{_RECO}/full_pvtag.parquet",
    log:
        f"{_RECO}/full_pvtag.log",
    params:
        data=config["input"],
        max_events=events("max_events_pvtag"),
    threads: workflow.cores
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --mode full --pvtag"
        " --input '{params.data}'"
        " --max-events {params.max_events}"
        " --workers {threads}"
        " --model {input.model}"
        " --out-file {output}"
        " > {log} 2>&1"


rule track_pv:
    input:
        script="physics/reconstruction/track_pv.py",
    output:
        f"{_RECO}/pv_assoc.parquet",
    log:
        f"{_RECO}/pv_assoc.log",
    params:
        data=config["input_minbias"],
        max_events=events("max_events_track_pv"),
        outdir=_RECO,
    threads: workflow.cores
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --input '{params.data}'"
        " --max-events {params.max_events}"
        " --workers {threads}"
        " --out-dir {params.outdir}"
        " > {log} 2>&1"


rule two_track_mva:
    input:
        model="models/two_track_mva.onnx",
        script="physics/reconstruction/two_track_mva.py",
    output:
        f"{_RECO}/mva.parquet",
    log:
        f"{_RECO}/mva.log",
    params:
        data=config["input"],
        max_events=events("max_events_two_track_mva"),
        outdir=_RECO,
    threads: workflow.cores
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --input '{params.data}'"
        " --max-events {params.max_events}"
        " --workers {threads}"
        " --model {input.model}"
        " --out-dir {params.outdir}"
        " > {log} 2>&1"


rule mumu_decay_performance:
    input:
        cheated=_cheated(events("max_events_full")),
        full=f"{_RECO}/full.parquet",
        script="physics/analysis/decay_performance_study/decay_performance.py",
    output:
        f"{_perf('')}/bs_eff_vs_kinematics.png",
        f"{_perf('')}/bs_mass.png",
    log:
        f"{_perf('')}/run.log",
    params:
        outdir=_perf(""),
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --cheated {input.cheated} --full {input.full}"
        " --channel bs_to_mumu"
        " --out-dir {params.outdir}"
        " > {log} 2>&1"


rule mumu_decay_performance_notime:
    input:
        cheated=_cheated(events("max_events_full_notime")),
        full=f"{_RECO}/full_notime.parquet",
        script="physics/analysis/decay_performance_study/decay_performance.py",
    output:
        f"{_perf('_notime')}/bs_eff_vs_kinematics.png",
        f"{_perf('_notime')}/bs_mass.png",
    log:
        f"{_perf('_notime')}/run.log",
    params:
        outdir=_perf("_notime"),
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --cheated {input.cheated} --full {input.full}"
        " --channel bs_to_mumu"
        " --tag 'NO TIMING'"
        " --out-dir {params.outdir}"
        " > {log} 2>&1"


rule mumu_decay_performance_pvtag:
    input:
        cheated=_cheated(events("max_events_pvtag")),
        full=f"{_RECO}/full_pvtag.parquet",
        script="physics/analysis/decay_performance_study/decay_performance.py",
    output:
        f"{_perf('_pvtag')}/bs_eff_vs_kinematics.png",
        f"{_perf('_pvtag')}/bs_mass.png",
    log:
        f"{_perf('_pvtag')}/run.log",
    params:
        outdir=_perf("_pvtag"),
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --cheated {input.cheated} --full {input.full}"
        " --channel bs_to_mumu"
        " --tag 'PV TAG'"
        " --out-dir {params.outdir}"
        " > {log} 2>&1"


rule pv_association:
    input:
        full=f"{_RECO}/full.parquet",
        notime=f"{_RECO}/full_notime.parquet",
        pvtag=f"{_RECO}/full_pvtag.parquet",
        script=(
            "physics/analysis/pv_association_study/" "pv_association_correctness.py"
        ),
    output:
        outdir("analysis", "pv_association") + "/pv_association_correctness.png",
    log:
        outdir("analysis", "pv_association") + "/run.log",
    params:
        datadir=_RECO,
        outdir=outdir("analysis", "pv_association"),
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --data-dir {params.datadir}"
        " --out-dir {params.outdir}"
        " > {log} 2>&1"


rule track_pv_timing_scan:
    input:
        assoc=f"{_RECO}/pv_assoc.parquet",
        script="physics/analysis/pv_association_study/track_pv_timing_scan.py",
    output:
        outdir("analysis", "track_pv_timing") + "/track_pv_timing_scan.png",
    log:
        outdir("analysis", "track_pv_timing") + "/run.log",
    params:
        indir=_RECO,
        outdir=outdir("analysis", "track_pv_timing"),
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --input-dir {params.indir}"
        " --out-dir {params.outdir}"
        " > {log} 2>&1"


rule sb_ratio_vs_eta:
    input:
        full=f"{_RECO}/full.parquet",
        script="physics/analysis/sb_ratio_study/sb_ratio_vs_eta.py",
    output:
        outdir("analysis", "sb_ratio") + "/sb_ratio_vs_eta.png",
        outdir("analysis", "sb_ratio") + "/mass_vs_eta.png",
    log:
        outdir("analysis", "sb_ratio") + "/run.log",
    params:
        outdir=outdir("analysis", "sb_ratio"),
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --full {input.full}"
        " --out-dir {params.outdir}"
        " > {log} 2>&1"


rule mumu_truth_distributions:
    input:
        cheated=_cheated(events("max_events_full")),
        script=("physics/analysis/truth_distribution_study/signal_distributions.py"),
    output:
        outdir("analysis", "truth_distributions") + "/bs_signal_distributions.png",
    log:
        outdir("analysis", "truth_distributions") + "/run.log",
    params:
        outdir=outdir("analysis", "truth_distributions"),
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --cheated {input.cheated}"
        " --channel bs_to_mumu"
        " --out-dir {params.outdir}"
        " > {log} 2>&1"


rule mumu_observables:
    input:
        dist=f"{_RECO}/dist.parquet",
        script=(
            "physics/analysis/reconstructed_distribution_study/"
            "signal_vs_background.py"
        ),
    output:
        outdir("analysis", "reconstructed_distributions") + "/bs_observables.png",
    log:
        outdir("analysis", "reconstructed_distributions") + "/run.log",
    params:
        outdir=outdir("analysis", "reconstructed_distributions"),
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --dist {input.dist}"
        " --channel bs_to_mumu"
        " --out-dir {params.outdir}"
        " > {log} 2>&1"


rule mva_performance:
    input:
        data=f"{_RECO}/mva.parquet",
        script="physics/analysis/two_track_mva_study/mva_performance.py",
    output:
        outdir("analysis", "two_track_mva") + "/mva_response_distribution.png",
        outdir("analysis", "two_track_mva") + "/tagged_pvs_per_event.png",
    log:
        outdir("analysis", "two_track_mva") + "/run.log",
    params:
        outdir=outdir("analysis", "two_track_mva"),
    shell:
        "PYTHONPATH=src python3 {input.script}"
        " --input {input.data}"
        " --channel bs_to_mumu"
        " --out-dir {params.outdir}"
        " > {log} 2>&1"
