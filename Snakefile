configfile: "config.yaml"


include: "workflow/reconstruction.smk"
include: "workflow/analysis/truth_distribution_study.smk"
include: "workflow/analysis/reconstructed_distribution_study.smk"
include: "workflow/analysis/decay_performance_study.smk"
include: "workflow/analysis/sb_ratio_study.smk"
include: "workflow/analysis/two_track_mva_study.smk"
include: "workflow/analysis/pv_association_study.smk"
include: "workflow/toy_study/vertex_fit_4d_vs_3d.smk"


rule all:
    input:
        rules.all_reconstruction.input,
        rules.all_truth_distribution_study.input,
        rules.all_reconstructed_distribution_study.input,
        rules.all_decay_performance_study.input,
        rules.all_sb_ratio_study.input,
        rules.all_two_track_mva_study.input,
        rules.all_pv_association_study.input,
        rules.all_vertex_fit_4d_vs_3d.input,
