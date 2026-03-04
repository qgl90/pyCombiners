LUMIS = list(config["luminosities"].keys())
KS_MODES = ["cheated", "full", "dist"]
BS_MODES = ["cheated", "full", "full_notime", "dist"]

wildcard_constraints:
    ks_mode="cheated|full|dist",
    bs_mode="cheated|full|full_notime|dist",


rule all_reconstruct:
    input:
        expand("public/{lumi}/reconstruct/ks_to_pipi/{mode}_summary.json",
               lumi=LUMIS, mode=KS_MODES),
        expand("public/{lumi}/reconstruct/bs_to_mumu/{mode}_summary.json",
               lumi=LUMIS, mode=BS_MODES),
        expand("public/{lumi}/reconstruct/track_pv_association/summary.json",
               lumi=LUMIS),


rule reconstruct_ks:
    input:
        lambda wc: config["luminosities"][wc.lumi]["channels"]["ks_to_pipi"]["modes"][wc.ks_mode].get(
            "input", config["luminosities"][wc.lumi]["channels"]["ks_to_pipi"]["input"]),
    output:
        "public/{lumi}/reconstruct/ks_to_pipi/{ks_mode}.parquet",
        "public/{lumi}/reconstruct/ks_to_pipi/{ks_mode}_summary.json",
    log:
        "public/{lumi}/reconstruct/ks_to_pipi/{ks_mode}.log",
    params:
        max_events=lambda wc: config["luminosities"][wc.lumi]["channels"]["ks_to_pipi"]["modes"][wc.ks_mode]["max_events"],
        tree=config["tree"],
    shell:
        "PYTHONPATH=src python3 physics/reconstruct/ks_to_pipi.py"
        " --mode {wildcards.ks_mode}"
        " --input {input} --tree {params.tree}"
        " --max-events {params.max_events}"
        " --out-dir public/{wildcards.lumi}/reconstruct/ks_to_pipi"
        " > {log} 2>&1"


rule reconstruct_bs:
    input:
        lambda wc: config["luminosities"][wc.lumi]["channels"]["bs_to_mumu"]["modes"][wc.bs_mode].get(
            "input", config["luminosities"][wc.lumi]["channels"]["bs_to_mumu"]["input"]),
    output:
        "public/{lumi}/reconstruct/bs_to_mumu/{bs_mode}.parquet",
        "public/{lumi}/reconstruct/bs_to_mumu/{bs_mode}_summary.json",
    log:
        "public/{lumi}/reconstruct/bs_to_mumu/{bs_mode}.log",
    params:
        max_events=lambda wc: config["luminosities"][wc.lumi]["channels"]["bs_to_mumu"]["modes"][wc.bs_mode]["max_events"],
        tree=config["tree"],
    shell:
        "PYTHONPATH=src python3 physics/reconstruct/bs_to_mumu.py"
        " --mode {wildcards.bs_mode}"
        " --input {input} --tree {params.tree}"
        " --max-events {params.max_events}"
        " --out-dir public/{wildcards.lumi}/reconstruct/bs_to_mumu"
        " > {log} 2>&1"


rule reconstruct_pv:
    input:
        lambda wc: config["luminosities"][wc.lumi]["channels"]["track_pv_association"]["input"],
    output:
        "public/{lumi}/reconstruct/track_pv_association/pv_stats.parquet",
        "public/{lumi}/reconstruct/track_pv_association/pv_assoc.parquet",
        "public/{lumi}/reconstruct/track_pv_association/summary.json",
    log:
        "public/{lumi}/reconstruct/track_pv_association/run.log",
    params:
        max_events=lambda wc: config["luminosities"][wc.lumi]["channels"]["track_pv_association"]["max_events"],
        tree=config["tree"],
    shell:
        "PYTHONPATH=src python3 physics/reconstruct/track_pv.py"
        " --input {input} --tree {params.tree}"
        " --max-events {params.max_events}"
        " --out-dir public/{wildcards.lumi}/reconstruct/track_pv_association"
        " > {log} 2>&1"
