#!/bin/bash
set -ex

# parameters: base_name termini_cleavage epitope_cleavage coverage
function run_if_new {
    if [ ! -f "$1.log" ]; then
        python design.py simple dev/made-epitopes.csv "$1.csv" \
            --max-spacer-length 4 \
            --min-spacer-length 4 \
            --num-epitopes 15 \
            --log-file "$1.log" \
            --min-nterminus-cleavage $2 \
            --min-cterminus-cleavage $2 \
            --epitope-cleavage-ignore-first 3 \
            --max-epitope-cleavage $3 \
            --min-proteins $4 \
            --verbose
    fi
}

run_if_new "dev/res-cov-1.74-2.0-1.0" "1.74" "2.0" "1.0"
run_if_new "dev/res-cov-1.74-1.0-1.0" "1.74" "1.0" "1.0"
run_if_new "dev/res-cov-1.74-0.5-1.0" "1.74" "0.5" "1.0"

run_if_new "dev/res-cov-1.95-2.0-1.0" "1.95" "2.0" "1.0"
run_if_new "dev/res-cov-1.95-1.0-1.0" "1.95" "1.0" "1.0"
run_if_new "dev/res-cov-1.95-0.5-1.0" "1.95" "0.5" "1.0"

run_if_new "dev/res-cov-2.5-2.0-1.0" "2.5" "2.0" "1.0"
run_if_new "dev/res-cov-2.5-1.0-1.0" "2.5" "1.0" "1.0"
run_if_new "dev/res-cov-2.5-0.5-1.0" "2.5" "0.5" "1.0"

