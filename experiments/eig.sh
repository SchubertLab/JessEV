#!/bin/bash
set -ex

# parameters: termini_cleavage epitope_cleavage spacer_cleavage
function run_if_new {
    basename="dev/res-eig-$1-$2-$3"
    if [ ! -f "$basename.log" ]; then
        python design.py simple dev/made-epitopes.csv "$basename.csv" \
            --log-file "$basename.log" --verbose \
            --num-epitopes 15 \
            --max-spacer-length 4 \
            --min-spacer-length 4 \
            --epitope-cleavage-ignore-first 3 \
            --min-nterminus-cleavage $1 \
            --min-cterminus-cleavage $1 \
            --max-epitope-cleavage $2 \
            --max-spacer-cleavage $3
    fi
}

for term in 1.5 2.0 2.5
do
    for inn in 1.0 0.0 -1.0
    do
        for spac in -1.0 0.0 1.0
        do
            run_if_new $term $inn $cov $cons $spac
        done
    done
done
