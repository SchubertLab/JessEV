set -x

ntc_grid="3.5 3.25 3 2.75 2.5 2.25 1.1 1.4 1.61 1.74 1.8 1.95"
epi_grid="-1.0 -0.5 -0.2 -0.1 0.0 0.1 0.2 0.5 1.0"

# parameters: termini_cleavage epitope_cleavage
function run_if_new {
    res="dev/res-comb-nc-$ntc-$epi.csv"
    log="dev/res-comb-nc-$ntc-$epi.log"
    if [ ! -f $log ]; then
        python design.py dev/made-epitopes.csv $res \
            --max-spacer-length 4 \
            --min-spacer-length 0 \
            --num-epitopes 5 \
            --min-nterminus-cleavage $1 \
            --min-cterminus-cleavage $1 \
            --max-epitope-cleavage $2 \
            --epitope-cleavage-ignore-first 4 \
            --log-file $log \
            --verbose
    fi
}


for ntc in $ntc_grid
do
    for epi in $epi_grid
    do
        run_if_new $ntc $epi
    done
done

