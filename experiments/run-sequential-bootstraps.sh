#!/bin/bash
set -ex

# parameters: epitopes_set_number
function run_if_new {
    basename="dev/res-sequential-set-$1"
    if [ ! -f "$basename.log" ]; then
        python run_sequential.py \
            "dev/epitopes-5k-set-$1.csv" \
            "dev/made-alleles.csv" \
            "dev/full-affinities.csv" \
            "$basename.csv" \
            --num-epitopes 5 --log-file "$basename.log" --verbose
    fi
}

for i in {1..30}; do run_if_new $i; done
