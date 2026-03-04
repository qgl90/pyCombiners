LUMIS = list(config["luminosities"].keys())


rule all_performance:
    input:
        expand("public/{lumi}/performance_study/ks_to_pipi/ks_eff_vs_kinematics.png", lumi=LUMIS),
        expand("public/{lumi}/performance_study/bs_to_mumu/bs_eff_vs_kinematics.png", lumi=LUMIS),
        expand("public/{lumi}/performance_study/bs_to_mumu_notime/bs_eff_vs_kinematics.png", lumi=LUMIS),


rule perf_ks:
    input:
        cheated="public/{lumi}/reconstruct/ks_to_pipi/cheated.parquet",
        full="public/{lumi}/reconstruct/ks_to_pipi/full.parquet",
    output:
        "public/{lumi}/performance_study/ks_to_pipi/ks_eff_vs_kinematics.png",
        "public/{lumi}/performance_study/ks_to_pipi/ks_mass.png",
    log:
        "public/{lumi}/performance_study/ks_to_pipi/run.log",
    shell:
        "PYTHONPATH=src python3 physics/analyze/perf_ks.py"
        " --cheated {input.cheated} --full {input.full}"
        " --out-dir public/{wildcards.lumi}/performance_study/ks_to_pipi"
        " --lumi {wildcards.lumi}"
        " > {log} 2>&1"


rule perf_bs:
    input:
        cheated="public/{lumi}/reconstruct/bs_to_mumu/cheated.parquet",
        full="public/{lumi}/reconstruct/bs_to_mumu/full.parquet",
    output:
        "public/{lumi}/performance_study/bs_to_mumu/bs_eff_vs_kinematics.png",
        "public/{lumi}/performance_study/bs_to_mumu/bs_mass.png",
    log:
        "public/{lumi}/performance_study/bs_to_mumu/run.log",
    shell:
        "PYTHONPATH=src python3 physics/analyze/perf_bs.py"
        " --cheated {input.cheated} --full {input.full}"
        " --out-dir public/{wildcards.lumi}/performance_study/bs_to_mumu"
        " --lumi {wildcards.lumi}"
        " > {log} 2>&1"


rule perf_bs_notime:
    input:
        cheated="public/{lumi}/reconstruct/bs_to_mumu/cheated.parquet",
        full_notime="public/{lumi}/reconstruct/bs_to_mumu/full_notime.parquet",
    output:
        "public/{lumi}/performance_study/bs_to_mumu_notime/bs_eff_vs_kinematics.png",
        "public/{lumi}/performance_study/bs_to_mumu_notime/bs_mass.png",
    log:
        "public/{lumi}/performance_study/bs_to_mumu_notime/run.log",
    shell:
        "PYTHONPATH=src python3 physics/analyze/perf_bs_notime.py"
        " --cheated {input.cheated} --full {input.full_notime}"
        " --out-dir public/{wildcards.lumi}/performance_study/bs_to_mumu_notime"
        " --lumi {wildcards.lumi}"
        " > {log} 2>&1"
