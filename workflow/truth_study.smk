LUMIS = list(config["luminosities"].keys())


rule all_truth:
    input:
        expand("public/{lumi}/truth_study/ks_to_pipi/ks_signal_distributions.png", lumi=LUMIS),
        expand("public/{lumi}/truth_study/bs_to_mumu/bs_signal_distributions.png", lumi=LUMIS),
        expand("public/{lumi}/truth_study/track_pv_association/track_pv_association.png", lumi=LUMIS),


rule truth_ks:
    input:
        cheated="public/{lumi}/reconstruct/ks_to_pipi/cheated.parquet",
        summary="public/{lumi}/reconstruct/ks_to_pipi/cheated_summary.json",
    output:
        "public/{lumi}/truth_study/ks_to_pipi/ks_signal_distributions.png",
    log:
        "public/{lumi}/truth_study/ks_to_pipi/run.log",
    shell:
        "PYTHONPATH=src python3 physics/analyze/truth_ks.py"
        " --cheated {input.cheated} --summary {input.summary}"
        " --out-dir public/{wildcards.lumi}/truth_study/ks_to_pipi"
        " --lumi {wildcards.lumi}"
        " > {log} 2>&1"


rule truth_bs:
    input:
        cheated="public/{lumi}/reconstruct/bs_to_mumu/cheated.parquet",
        summary="public/{lumi}/reconstruct/bs_to_mumu/cheated_summary.json",
    output:
        "public/{lumi}/truth_study/bs_to_mumu/bs_signal_distributions.png",
    log:
        "public/{lumi}/truth_study/bs_to_mumu/run.log",
    shell:
        "PYTHONPATH=src python3 physics/analyze/truth_bs.py"
        " --cheated {input.cheated} --summary {input.summary}"
        " --out-dir public/{wildcards.lumi}/truth_study/bs_to_mumu"
        " --lumi {wildcards.lumi}"
        " > {log} 2>&1"


rule truth_pv:
    input:
        stats="public/{lumi}/reconstruct/track_pv_association/pv_stats.parquet",
        assoc="public/{lumi}/reconstruct/track_pv_association/pv_assoc.parquet",
        summary="public/{lumi}/reconstruct/track_pv_association/summary.json",
    output:
        "public/{lumi}/truth_study/track_pv_association/track_pv_association.png",
        "public/{lumi}/truth_study/track_pv_association/pv_assoc_efficiency.png",
    log:
        "public/{lumi}/truth_study/track_pv_association/run.log",
    shell:
        "PYTHONPATH=src python3 physics/analyze/truth_pv.py"
        " --input-dir public/{wildcards.lumi}/reconstruct/track_pv_association"
        " --out-dir public/{wildcards.lumi}/truth_study/track_pv_association"
        " --lumi {wildcards.lumi}"
        " > {log} 2>&1"
