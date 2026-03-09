#!/bin/bash
# Sync ntuple_*_1000evts.root files to EOS with auto-retry support

SRC="input/"
DST="/eos/lhcb/user/j/jzhuo/pyCombiners/input/v0_2_0/"
MAX_RETRIES=${1:-5}  # default 5, or pass as first argument

# Create destination directory if it doesn't exist
mkdir -p "$DST"

for attempt in $(seq 1 "$MAX_RETRIES"); do
    echo "=== Attempt $attempt / $MAX_RETRIES ==="
    # rsync with:
    #   --partial        : keep partially transferred files (resume support)
    #   --progress       : show transfer progress
    #   --include/exclude: only copy ntuple_*_1000evts.root files
    #   -v               : verbose
    rsync -rv --partial --progress \
        --include='ntuple_*_1000evts.root' \
        --exclude='*' \
        "$SRC" "$DST"

    if [ $? -eq 0 ]; then
        echo ""
        echo "All files synced successfully on attempt $attempt."
        exit 0
    fi

    echo "Attempt $attempt failed."
    if [ "$attempt" -lt "$MAX_RETRIES" ]; then
        echo "Retrying in 5 seconds..."
        sleep 5
    fi
done

echo ""
echo "Failed after $MAX_RETRIES attempts."
exit 1
