#!/bin/bash
set -ex

# parameters: epitopes_set_number
function run_if_new {
    basename="dev/res-boostrap-ours-set-$1"
    if [ ! -f "$basename.log" ]; then
        python design.py simple \
            "dev/epitopes-5k-set$1.csv" \
            "$basename.csv" \
            --min-spacer-length 4 \
            --max-spacer-length 4 \
            --num-epitopes 5 \
            --min-cterminus-cleavage 2 \
            --min-nterminus-cleavage 2 \
            --max-spacer-cleavage -1 \
            --max-epitope-cleavage -1 \
            --epitope-cleavage-ignore-first 3 \
            --log-file "$basename.log" --verbose
    fi
}

run_if_new 1
run_if_new 2
run_if_new 3
run_if_new 4
run_if_new 5
