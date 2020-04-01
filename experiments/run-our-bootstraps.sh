#!/bin/bash
set -ex

# this script runs our framework on the thirty bootstraps

# parameters: epitopes_set_number max_epitope_cleavage
function run_if_new {
    basename="dev/res-boostrap-ours-set-$1-$2"
    if [ ! -f "$basename.log" ]; then
        python design.py \
            "dev/epitopes-5k-set-$1.csv" \
            "$basename.csv" \
            --min-spacer-length 4 \
            --max-spacer-length 4 \
            --num-epitopes 5 \
            --min-cterminus-cleavage 2 \
            --min-nterminus-cleavage 2 \
            --max-spacer-cleavage -2 \
            --max-epitope-cleavage $2 \
            --epitope-cleavage-ignore-first 3 \
            --log-file "$basename.log" --verbose
    fi
}

for i in {1..30}
do 
    run_if_new $i 0
    run_if_new $i -0.5
    run_if_new $i -1
done
