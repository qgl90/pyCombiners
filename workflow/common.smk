# Shared helpers for the per-pipeline snakefiles.
# Each pipeline loads its own config/<name>.yaml. Override ad hoc with
# --config max_events=1000 output_dir=public/test, or pass a custom
# file with --configfile (e.g. another sample / luminosity).
# Reconstruction rules take all cores (--workers); analyses run on one
# core each so snakemake can run several of them in parallel.


def outdir(*parts):
    return "/".join([config["output_dir"], *parts])


def events(key):
    return config.get("max_events", config[key])
