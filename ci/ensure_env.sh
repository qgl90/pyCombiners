#!/bin/bash
# Recreate the CI env on the runner VM when environment.yaml changes.
set -e
mm=/data/micromamba/bin/micromamba
export MAMBA_ROOT_PREFIX=/data/micromamba
stamp=/data/micromamba/run5.stamp
new=$(sha256sum environment.yaml | cut -d' ' -f1)
if [ ! -f "$stamp" ] || [ "$(cat "$stamp")" != "$new" ]; then
    "$mm" create -y -n run5 -f environment.yaml
    echo "$new" > "$stamp"
fi
