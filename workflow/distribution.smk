LUMIS = list(config["luminosities"].keys())


rule all_distribution:
    input:
        expand("public/{lumi}/distribution_study/ks_to_pipi/ks_observables.png", lumi=LUMIS),
        expand("public/{lumi}/distribution_study/bs_to_mumu/bs_observables.png", lumi=LUMIS),


rule dist_ks:
    input:
        dist="public/{lumi}/reconstruct/ks_to_pipi/dist.parquet",
        summary="public/{lumi}/reconstruct/ks_to_pipi/dist_summary.json",
    output:
        "public/{lumi}/distribution_study/ks_to_pipi/ks_observables.png",
    log:
        "public/{lumi}/distribution_study/ks_to_pipi/run.log",
    shell:
        "PYTHONPATH=src python3 physics/analyze/dist_ks.py"
        " --dist {input.dist} --summary {input.summary}"
        " --out-dir public/{wildcards.lumi}/distribution_study/ks_to_pipi"
        " --lumi {wildcards.lumi}"
        " > {log} 2>&1"


rule dist_bs:
    input:
        dist="public/{lumi}/reconstruct/bs_to_mumu/dist.parquet",
        summary="public/{lumi}/reconstruct/bs_to_mumu/dist_summary.json",
    output:
        "public/{lumi}/distribution_study/bs_to_mumu/bs_observables.png",
    log:
        "public/{lumi}/distribution_study/bs_to_mumu/run.log",
    shell:
        "PYTHONPATH=src python3 physics/analyze/dist_bs.py"
        " --dist {input.dist} --summary {input.summary}"
        " --out-dir public/{wildcards.lumi}/distribution_study/bs_to_mumu"
        " --lumi {wildcards.lumi}"
        " > {log} 2>&1"
