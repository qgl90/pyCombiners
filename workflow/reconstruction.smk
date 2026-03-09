_OUTDIR = config["output_dir"]
_LUMIS = list(config["luminosities"].keys())
KS_RECO_MODES = ["full", "dist"]
BS_RECO_MODES = ["full", "full_notime", "dist"]

# --- Config validation ---
for _lumi in _LUMIS:
    _ch = config["luminosities"][_lumi]["channels"]
    for _name in ["track_pv_association", "two_track_mva", "bs_to_mumu_pvtag"]:
        assert (
            _name in _ch and "input" in _ch[_name] and "max_events" in _ch[_name]
        ), f"config: {_lumi}/{_name} missing or incomplete"
    for _name, _modes in [
        ("ks_to_pipi", ["cheated"] + KS_RECO_MODES),
        ("bs_to_mumu", ["cheated"] + BS_RECO_MODES),
        ("bs_to_jpsiphi", ["cheated", "full"]),
    ]:
        assert (
            _name in _ch and "input" in _ch[_name]
        ), f"config: {_lumi}/{_name} missing or incomplete"
        for _m in _modes:
            assert _m in _ch[_name].get(
                "modes", {}
            ), f"config: {_lumi}/{_name} missing mode '{_m}'"


wildcard_constraints:
    ks_mode="full|dist",
    bs_mode="full|full_notime|dist",
    n_events="[0-9]+",


# Collect unique ks cheated n_events per lumi
def _ks_cheated_parquets():
    paths = []
    for lumi in _LUMIS:
        seen = set()
        modes = config["luminosities"][lumi]["channels"]["ks_to_pipi"]["modes"]
        for m in ["cheated"] + KS_RECO_MODES:
            n = modes[m]["max_events"]
            if n not in seen:
                seen.add(n)
                paths.append(
                    f"{_OUTDIR}/{lumi}/reconstruction/ks_to_pipi/cheated_{n}.parquet"
                )
    return paths


# Collect unique bs cheated n_events per lumi
def _bs_cheated_parquets():
    paths = []
    for lumi in _LUMIS:
        seen = set()
        modes = config["luminosities"][lumi]["channels"]["bs_to_mumu"]["modes"]
        for m in ["cheated"] + BS_RECO_MODES:
            n = modes[m]["max_events"]
            if n not in seen:
                seen.add(n)
                paths.append(
                    f"{_OUTDIR}/{lumi}/reconstruction/bs_to_mumu/cheated_{n}.parquet"
                )
        # pvtag also needs a cheated reference
        n = config["luminosities"][lumi]["channels"]["bs_to_mumu_pvtag"]["max_events"]
        if n not in seen:
            seen.add(n)
            paths.append(
                f"{_OUTDIR}/{lumi}/reconstruction/bs_to_mumu/cheated_{n}.parquet"
            )
    return paths


def _jpsiphi_cheated_parquets():
    paths = []
    for lumi in _LUMIS:
        seen = set()
        modes = config["luminosities"][lumi]["channels"]["bs_to_jpsiphi"]["modes"]
        for m in ["cheated", "full"]:
            n = modes[m]["max_events"]
            if n not in seen:
                seen.add(n)
                paths.append(
                    f"{_OUTDIR}/{lumi}/reconstruction/bs_to_jpsiphi/cheated_{n}.parquet"
                )
    return paths


rule all_reconstruction:
    input:
        _ks_cheated_parquets(),
        expand(
            f"{_OUTDIR}/{{lumi}}/reconstruction/ks_to_pipi/{{mode}}.parquet",
            lumi=_LUMIS,
            mode=KS_RECO_MODES,
        ),
        _bs_cheated_parquets(),
        expand(
            f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/{{mode}}.parquet",
            lumi=_LUMIS,
            mode=BS_RECO_MODES,
        ),
        _jpsiphi_cheated_parquets(),
        expand(
            f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_jpsiphi/full.parquet",
            lumi=_LUMIS,
        ),
        expand(
            f"{_OUTDIR}/{{lumi}}/reconstruction/track_pv_association/pv_assoc.parquet",
            lumi=_LUMIS,
        ),
        expand(
            f"{_OUTDIR}/{{lumi}}/reconstruction/two_track_mva/mva.parquet", lumi=_LUMIS
        ),
        expand(
            f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/full_pvtag.parquet",
            lumi=_LUMIS,
        ),


rule reconstruction_ks_cheated:
    input:
        data=lambda wc: config["luminosities"][wc.lumi]["channels"]["ks_to_pipi"][
            "input"
        ],
        script="physics/reconstruction/ks_to_pipi.py",
    output:
        f"{_OUTDIR}/{{lumi}}/reconstruction/ks_to_pipi/cheated_{{n_events}}.parquet",
    log:
        f"{_OUTDIR}/{{lumi}}/reconstruction/ks_to_pipi/cheated_{{n_events}}.log",
    params:
        tree=config["tree"],
        outdir=lambda wc: f"{config['output_dir']}/{wc.lumi}/reconstruction/ks_to_pipi",
    shell:
        "PYTHONPATH=src python3 physics/reconstruction/ks_to_pipi.py"
        " --mode cheated"
        " --input {input.data} --tree {params.tree}"
        " --max-events {wildcards.n_events}"
        " --out-dir {params.outdir}"
        " --out-file {output}"
        " > {log} 2>&1"


rule reconstruction_ks_to_pipi:
    input:
        data=lambda wc: config["luminosities"][wc.lumi]["channels"]["ks_to_pipi"][
            "modes"
        ][wc.ks_mode].get(
            "input", config["luminosities"][wc.lumi]["channels"]["ks_to_pipi"]["input"]
        ),
        cheated=lambda wc: f"{config['output_dir']}/{wc.lumi}/reconstruction/ks_to_pipi/cheated_{config['luminosities'][wc.lumi]['channels']['ks_to_pipi']['modes'][wc.ks_mode]['max_events']}.parquet",
        script="physics/reconstruction/ks_to_pipi.py",
    output:
        f"{_OUTDIR}/{{lumi}}/reconstruction/ks_to_pipi/{{ks_mode}}.parquet",
    log:
        f"{_OUTDIR}/{{lumi}}/reconstruction/ks_to_pipi/{{ks_mode}}.log",
    params:
        max_events=lambda wc: config["luminosities"][wc.lumi]["channels"][
            "ks_to_pipi"
        ]["modes"][wc.ks_mode]["max_events"],
        tree=config["tree"],
        outdir=_OUTDIR,
    shell:
        "PYTHONPATH=src python3 physics/reconstruction/ks_to_pipi.py"
        " --mode {wildcards.ks_mode}"
        " --input {input.data} --tree {params.tree}"
        " --max-events {params.max_events}"
        " --out-dir {params.outdir}/{wildcards.lumi}/reconstruction/ks_to_pipi"
        " > {log} 2>&1"


rule reconstruction_bs_cheated:
    input:
        data=lambda wc: config["luminosities"][wc.lumi]["channels"]["bs_to_mumu"][
            "input"
        ],
        script="physics/reconstruction/bs_to_mumu.py",
    output:
        f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/cheated_{{n_events}}.parquet",
    log:
        f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/cheated_{{n_events}}.log",
    params:
        tree=config["tree"],
        outdir=lambda wc: f"{config['output_dir']}/{wc.lumi}/reconstruction/bs_to_mumu",
    shell:
        "PYTHONPATH=src python3 physics/reconstruction/bs_to_mumu.py"
        " --mode cheated"
        " --input {input.data} --tree {params.tree}"
        " --max-events {wildcards.n_events}"
        " --out-dir {params.outdir}"
        " --out-file {output}"
        " > {log} 2>&1"


rule reconstruction_bs_to_mumu:
    input:
        data=lambda wc: config["luminosities"][wc.lumi]["channels"]["bs_to_mumu"][
            "modes"
        ][wc.bs_mode].get(
            "input", config["luminosities"][wc.lumi]["channels"]["bs_to_mumu"]["input"]
        ),
        cheated=lambda wc: f"{config['output_dir']}/{wc.lumi}/reconstruction/bs_to_mumu/cheated_{config['luminosities'][wc.lumi]['channels']['bs_to_mumu']['modes'][wc.bs_mode]['max_events']}.parquet",
        script="physics/reconstruction/bs_to_mumu.py",
    output:
        f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/{{bs_mode}}.parquet",
    log:
        f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/{{bs_mode}}.log",
    params:
        max_events=lambda wc: config["luminosities"][wc.lumi]["channels"][
            "bs_to_mumu"
        ]["modes"][wc.bs_mode]["max_events"],
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
        data=lambda wc: config["luminosities"][wc.lumi]["channels"][
            "track_pv_association"
        ]["input"],
        script="physics/reconstruction/track_pv.py",
    output:
        f"{_OUTDIR}/{{lumi}}/reconstruction/track_pv_association/pv_assoc.parquet",
    log:
        f"{_OUTDIR}/{{lumi}}/reconstruction/track_pv_association/run.log",
    params:
        max_events=lambda wc: config["luminosities"][wc.lumi]["channels"][
            "track_pv_association"
        ]["max_events"],
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
        data=lambda wc: config["luminosities"][wc.lumi]["channels"]["two_track_mva"][
            "input"
        ],
        model="models/two_track_mva.onnx",
        script="physics/reconstruction/two_track_mva.py",
    output:
        f"{_OUTDIR}/{{lumi}}/reconstruction/two_track_mva/mva.parquet",
    log:
        f"{_OUTDIR}/{{lumi}}/reconstruction/two_track_mva/run.log",
    params:
        max_events=lambda wc: config["luminosities"][wc.lumi]["channels"][
            "two_track_mva"
        ]["max_events"],
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
        data=lambda wc: config["luminosities"][wc.lumi]["channels"][
            "bs_to_mumu_pvtag"
        ]["input"],
        model="models/two_track_mva.onnx",
        cheated=lambda wc: f"{config['output_dir']}/{wc.lumi}/reconstruction/bs_to_mumu/cheated_{config['luminosities'][wc.lumi]['channels']['bs_to_mumu_pvtag']['max_events']}.parquet",
        script="physics/reconstruction/bs_to_mumu.py",
    output:
        f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/full_pvtag.parquet",
    log:
        f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_mumu/full_pvtag.log",
    params:
        max_events=lambda wc: config["luminosities"][wc.lumi]["channels"][
            "bs_to_mumu_pvtag"
        ]["max_events"],
        tree=config["tree"],
    shell:
        "PYTHONPATH=src python3 physics/reconstruction/bs_to_mumu.py"
        " --mode full --pvtag"
        " --input {input.data} --tree {params.tree}"
        " --max-events {params.max_events}"
        " --model {input.model}"
        " --out-file {output}"
        " > {log} 2>&1"


rule reconstruction_jpsiphi_cheated:
    input:
        data=lambda wc: config["luminosities"][wc.lumi]["channels"]["bs_to_jpsiphi"][
            "input"
        ],
        script="physics/reconstruction/bs_to_jpsiphi.py",
    output:
        f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_jpsiphi/cheated_{{n_events}}.parquet",
    log:
        f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_jpsiphi/cheated_{{n_events}}.log",
    params:
        tree=config["tree"],
        outdir=lambda wc: f"{config['output_dir']}/{wc.lumi}/reconstruction/bs_to_jpsiphi",
    shell:
        "PYTHONPATH=src python3 physics/reconstruction/bs_to_jpsiphi.py"
        " --mode cheated"
        " --input {input.data} --tree {params.tree}"
        " --max-events {wildcards.n_events}"
        " --out-dir {params.outdir}"
        " --out-file {output}"
        " > {log} 2>&1"


rule reconstruction_jpsiphi_full:
    input:
        data=lambda wc: config["luminosities"][wc.lumi]["channels"]["bs_to_jpsiphi"][
            "modes"
        ]["full"].get(
            "input",
            config["luminosities"][wc.lumi]["channels"]["bs_to_jpsiphi"]["input"],
        ),
        cheated=lambda wc: f"{config['output_dir']}/{wc.lumi}/reconstruction/bs_to_jpsiphi/cheated_{config['luminosities'][wc.lumi]['channels']['bs_to_jpsiphi']['modes']['full']['max_events']}.parquet",
        script="physics/reconstruction/bs_to_jpsiphi.py",
    output:
        f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_jpsiphi/full.parquet",
    log:
        f"{_OUTDIR}/{{lumi}}/reconstruction/bs_to_jpsiphi/full.log",
    params:
        max_events=lambda wc: config["luminosities"][wc.lumi]["channels"][
            "bs_to_jpsiphi"
        ]["modes"]["full"]["max_events"],
        tree=config["tree"],
        outdir=_OUTDIR,
    shell:
        "PYTHONPATH=src python3 physics/reconstruction/bs_to_jpsiphi.py"
        " --mode full"
        " --input {input.data} --tree {params.tree}"
        " --max-events {params.max_events}"
        " --out-dir {params.outdir}/{wildcards.lumi}/reconstruction/bs_to_jpsiphi"
        " --out-file {output}"
        " > {log} 2>&1"
