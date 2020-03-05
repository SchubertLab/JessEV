#!/bin/bash
set -ex

# parameters: cleavage_prior
function run_if_new {
    basename="dev/res-mceig-$1"
    if [ ! -f "$basename.log" ]; then
        python seqdesign.py "$basename.csv" \
            --log-file "$basename.log" --verbose \
            --cleavage-prior $1
    fi
}

for prior in 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.75 1.0
do
    run_if_new $prior
done
