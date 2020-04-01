#!/bin/bash
set -ex

# this script runs a grid search over cleavage, coverage and conservation.
# the results are evaluated later to maximize the effective coverage

# parameters: termini_cleavage epitope_cleavage coverage conservation
function run_if_new {
    basename="dev/res-cov-$term-$inn-$cov-$cons"
    if [ ! -f "$basename.log" ]; then
        python design.py dev/made-epitopes.csv "$basename.csv" \
            --max-spacer-length 4 \
            --min-spacer-length 4 \
            --num-epitopes 15 \
            --log-file "$basename.log" \
            --epitope-cleavage-ignore-first 3 \
            --min-nterminus-cleavage $1 \
            --min-cterminus-cleavage $1 \
            --max-epitope-cleavage $2 \
            --min-proteins $3 \
    	    --min-avg-prot-conservation $4 \
            --verbose
    fi
}

for cons in 0.05 0.1 0.15 0.2
do
    for cov in 0.99
    do
        for term in 1.74 1.95 2.05 2.5
        do
            for inn in 2.0 1.5 1.0 0.0
            do
                run_if_new $term $inn $cov $cons
            done
        done
    done
done
