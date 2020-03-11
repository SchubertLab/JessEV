#!/bin/bash
set -ex

# parameters: epitopes_set_number
function run_if_new {
    basename="dev/res-sequential-set-$1"
    if [ ! -f "$basename.log" ]; then
        python run_sequential.py \
            "dev/epitopes-5k-set$1.csv" \
            "dev/made-alleles.csv" \
            "dev/full-affinities.csv" \
            "$basename.csv" \
            --num-epitopes 5 --log-file "$basename.log" --verbose
    fi
}

run_if_new 1
run_if_new 2
run_if_new 3
run_if_new 4
run_if_new 5
