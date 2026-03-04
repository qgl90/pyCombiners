configfile: "config.yaml"

include: "workflow/reconstruct.smk"
include: "workflow/truth_study.smk"
include: "workflow/distribution.smk"
include: "workflow/performance.smk"


rule all:
    input:
        rules.all_reconstruct.input,
        rules.all_truth.input,
        rules.all_distribution.input,
        rules.all_performance.input,
