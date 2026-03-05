_OUTDIR = config["output_dir"]
_LUMIS = list(config["luminosities"].keys())
KS_MODES = ["cheated", "full", "dist"]
BS_MODES = ["cheated", "full", "full_notime", "dist"]

# --- Config validation ---
for _lumi in _LUMIS:
    _ch = config["luminosities"][_lumi]["channels"]
    for _name in ["track_pv_association", "two_track_mva", "bs_to_mumu_pvtag"]:
        assert _name in _ch and "input" in _ch[_name] and "max_events" in _ch[_name], \
            f"config: {_lumi}/{_name} missing or incomplete"
    for _name, _modes in [("ks_to_pipi", KS_MODES), ("bs_to_mumu", BS_MODES)]:
        assert _name in _ch and "input" in _ch[_name], \
            f"config: {_lumi}/{_name} missing or incomplete"
        for _m in _modes:
            assert _m in _ch[_name].get("modes", {}), \
                f"config: {_lumi}/{_name} missing mode '{_m}'"

wildcard_constraints:
    ks_mode="cheated|full|dist",
    bs_mode="cheated|full|full_notime|dist",


rule all_reconstruction:
    input:
        expand(f"{_OUTDIR}/{{lumi}}/reconstruction/ks_to_pipi/{{mode}}.parquet",
               lumi=_LUMIS, mode=KS_MODES),
        expand(f"{_OUTDIR}/{{lumi}}/reconstruction/ks_to_pipi/{{mode}}_cheated.parquet",
               lumi=_LUMIS, mode=KS_MODES),
        expand(f"{_OUTDIR}/{{lumi}}/reconstruction/ks_to_pipi/{{mode}}_summary.json",
               lumi=_LUMIS, mode=KS_MODES),
        expand(f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/{{mode}}.parquet",
               lumi=_LUMIS, mode=BS_MODES),
        expand(f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/{{mode}}_cheated.parquet",
               lumi=_LUMIS, mode=BS_MODES),
        expand(f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/{{mode}}_summary.json",
               lumi=_LUMIS, mode=BS_MODES),
        expand(f"{_OUTDIR}/{{lumi}}/reconstruction/track_pv_association/pv_stats.parquet",
               lumi=_LUMIS),
        expand(f"{_OUTDIR}/{{lumi}}/reconstruction/track_pv_association/pv_assoc.parquet",
               lumi=_LUMIS),
        expand(f"{_OUTDIR}/{{lumi}}/reconstruction/track_pv_association/summary.json",
               lumi=_LUMIS),
        expand(f"{_OUTDIR}/{{lumi}}/reconstruction/two_track_mva/mva.parquet",
               lumi=_LUMIS),
        expand(f"{_OUTDIR}/{{lumi}}/reconstruction/two_track_mva/mva_summary.json",
               lumi=_LUMIS),
        expand(f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/pvtag.parquet",
               lumi=_LUMIS),
        expand(f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/pvtag_cheated.parquet",
               lumi=_LUMIS),
        expand(f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/pvtag_summary.json",
               lumi=_LUMIS),


rule reconstruction_ks_to_pipi:
    input:
        data=lambda wc: config["luminosities"][wc.lumi]["channels"]["ks_to_pipi"]["modes"][wc.ks_mode].get(
            "input", config["luminosities"][wc.lumi]["channels"]["ks_to_pipi"]["input"]),
        script="physics/reconstruction/ks_to_pipi.py",
    output:
        f"{_OUTDIR}/{{lumi}}/reconstruction/ks_to_pipi/{{ks_mode}}.parquet",
        f"{_OUTDIR}/{{lumi}}/reconstruction/ks_to_pipi/{{ks_mode}}_cheated.parquet",
        f"{_OUTDIR}/{{lumi}}/reconstruction/ks_to_pipi/{{ks_mode}}_summary.json",
    log:
        f"{_OUTDIR}/{{lumi}}/reconstruction/ks_to_pipi/{{ks_mode}}.log",
    params:
        max_events=lambda wc: config["luminosities"][wc.lumi]["channels"]["ks_to_pipi"]["modes"][wc.ks_mode]["max_events"],
        tree=config["tree"],
        outdir=_OUTDIR,
    shell:
        "PYTHONPATH=src python3 physics/reconstruction/ks_to_pipi.py"
        " --mode {wildcards.ks_mode}"
        " --input {input.data} --tree {params.tree}"
        " --max-events {params.max_events}"
        " --out-dir {params.outdir}/{wildcards.lumi}/reconstruction/ks_to_pipi"
        " > {log} 2>&1"


rule reconstruction_bs_to_mumu:
    input:
        data=lambda wc: config["luminosities"][wc.lumi]["channels"]["bs_to_mumu"]["modes"][wc.bs_mode].get(
            "input", config["luminosities"][wc.lumi]["channels"]["bs_to_mumu"]["input"]),
        script="physics/reconstruction/bs_to_mumu.py",
    output:
        f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/{{bs_mode}}.parquet",
        f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/{{bs_mode}}_cheated.parquet",
        f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/{{bs_mode}}_summary.json",
    log:
        f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/{{bs_mode}}.log",
    params:
        max_events=lambda wc: config["luminosities"][wc.lumi]["channels"]["bs_to_mumu"]["modes"][wc.bs_mode]["max_events"],
        tree=config["tree"],
        outdir=_OUTDIR,
    shell:
        "PYTHONPATH=src python3 physics/reconstruction/bs_to_mumu.py"
        " --mode {wildcards.bs_mode}"
        " --input {input.data} --tree {params.tree}"
        " --max-events {params.max_events}"
        " --out-dir {params.outdir}/{wildcards.lumi}/reconstruction/bs_to_mumu"
        " > {log} 2>&1"


rule reconstruction_track_pv:
    input:
        data=lambda wc: config["luminosities"][wc.lumi]["channels"]["track_pv_association"]["input"],
        script="physics/reconstruction/track_pv.py",
    output:
        f"{_OUTDIR}/{{lumi}}/reconstruction/track_pv_association/pv_stats.parquet",
        f"{_OUTDIR}/{{lumi}}/reconstruction/track_pv_association/pv_assoc.parquet",
        f"{_OUTDIR}/{{lumi}}/reconstruction/track_pv_association/summary.json",
    log:
        f"{_OUTDIR}/{{lumi}}/reconstruction/track_pv_association/run.log",
    params:
        max_events=lambda wc: config["luminosities"][wc.lumi]["channels"]["track_pv_association"]["max_events"],
        tree=config["tree"],
        outdir=_OUTDIR,
    shell:
        "PYTHONPATH=src python3 physics/reconstruction/track_pv.py"
        " --input {input.data} --tree {params.tree}"
        " --max-events {params.max_events}"
        " --out-dir {params.outdir}/{wildcards.lumi}/reconstruction/track_pv_association"
        " > {log} 2>&1"


rule reconstruction_two_track_mva:
    input:
        data=lambda wc: config["luminosities"][wc.lumi]["channels"]["two_track_mva"]["input"],
        model="models/two_track_mva.onnx",
        script="physics/reconstruction/two_track_mva.py",
    output:
        f"{_OUTDIR}/{{lumi}}/reconstruction/two_track_mva/mva.parquet",
        f"{_OUTDIR}/{{lumi}}/reconstruction/two_track_mva/mva_summary.json",
    log:
        f"{_OUTDIR}/{{lumi}}/reconstruction/two_track_mva/run.log",
    params:
        max_events=lambda wc: config["luminosities"][wc.lumi]["channels"]["two_track_mva"]["max_events"],
        tree=config["tree"],
        outdir=_OUTDIR,
    shell:
        "PYTHONPATH=src python3 physics/reconstruction/two_track_mva.py"
        " --input {input.data} --tree {params.tree}"
        " --max-events {params.max_events}"
        " --model {input.model}"
        " --out-dir {params.outdir}/{wildcards.lumi}/reconstruction/two_track_mva"
        " > {log} 2>&1"


rule reconstruction_bs_to_mumu_pvtag:
    input:
        data=lambda wc: config["luminosities"][wc.lumi]["channels"]["bs_to_mumu_pvtag"]["input"],
        model="models/two_track_mva.onnx",
        script="physics/reconstruction/bs_to_mumu_pvtag.py",
    output:
        f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/pvtag.parquet",
        f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/pvtag_cheated.parquet",
        f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/pvtag_summary.json",
    log:
        f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/pvtag.log",
    params:
        max_events=lambda wc: config["luminosities"][wc.lumi]["channels"]["bs_to_mumu_pvtag"]["max_events"],
        tree=config["tree"],
        outdir=_OUTDIR,
    shell:
        "PYTHONPATH=src python3 physics/reconstruction/bs_to_mumu_pvtag.py"
        " --input {input.data} --tree {params.tree}"
        " --max-events {params.max_events}"
        " --model {input.model}"
        " --out-dir {params.outdir}/{wildcards.lumi}/reconstruction/bs_to_mumu"
        " > {log} 2>&1"
