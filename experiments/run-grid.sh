set -x

ntc_grid="3.5 3.25 3 2.75 2.5 2.25 1.1 1.4 1.61 1.74 1.8 1.95"
spa_grid="0.0 0.1 0.2 0.5 0.75 1.0"
epi_grid_fine="-0.2 -0.15 -0.1 -0.05 0.0 0.05 0.1 0.15 0.2"
epi_grid_big="-1.0 -0.5 -0.2 -0.1 0.0 0.1 0.2 0.5 1.0"
epi_grid_small="-0.2 -0.1 0.0 0.1 0.2"

# remove incomplete results first (unfinished or errors)
grep -L "'result': {'completed': True, 'result':" dev/*.log | xargs rm



# do at least as badly as the sequential design
res="dev/match-full-sequential.csv"
log="dev/match-full-sequential.log"
if [ ! -f $log ]; then
    python design_strobe_spacers.py dev/full-epitopes.csv $res \
           --max-spacer-length 4 \
           --min-spacer-length 0 \
           --num-epitopes 5 \
           --log-file $log \
           --min-nterminus-cleavage 1.39 \
           --min-cterminus-cleavage 1.36 \
           --verbose
fi

# do at least as badly as the sequential design, controlling for epitope cleavage
res="dev/match-full-sequential-epi.csv"
log="dev/match-full-sequential-epi.log"
if [ ! -f $log ]; then
    python design_strobe_spacers.py dev/full-epitopes.csv $res \
           --max-spacer-length 4 \
           --min-spacer-length 0 \
           --num-epitopes 5 \
           --log-file $log \
           --min-nterminus-cleavage 1.39 \
           --min-cterminus-cleavage 1.36 \
           --epitope-cleavage-ignore-first 4 \
           --max-epitope-cleavage 0.0 \
           --verbose
fi

# now show a decent solution on the full set
res="dev/match-full-sequential-cleav2.csv"
log="dev/match-full-sequential-cleav2.log"
if [ ! -f $log ]; then
    python design_strobe_spacers.py dev/full-epitopes.csv $res \
           --max-spacer-length 4 \
           --min-spacer-length 0 \
           --num-epitopes 5 \
           --log-file $log \
           --min-nterminus-cleavage 2 \
           --min-cterminus-cleavage 2 \
           --epitope-cleavage-ignore-first 4 \
           --max-epitope-cleavage -0.1 \
           --verbose
fi


# do a grid search on n+c terminus cleavages
for cleav in 1.5 2 2.4 2.3 2.2 2.1 2.5 2.75 3
do
    res="dev/res-cleavages-$cleav.csv"
    log="dev/res-cleavages-$cleav.log"
    if [ ! -f $log ]; then
        python design_strobe_spacers.py dev/made-epitopes.csv $res \
            --max-spacer-length 4 \
            --min-spacer-length 0 \
            --num-epitopes 5 \
            --log-file $log \
            --min-nterminus-cleavage $cleav \
            --min-cterminus-cleavage $cleav \
            --epitope-cleavage-ignore-first 4 \
            --max-epitope-cleavage 0.0 \
            --verbose
    fi
done


# maximum immunogenicity without constraints
res="dev/res-maxig.csv"
log="dev/res-maxig.log"
if [ ! -f $log ]; then
    python design_strobe_spacers.py dev/made-epitopes.csv $res \
           --max-spacer-length 1 \
           --min-spacer-length 0 \
           --num-epitopes 5 \
           --log-file $log \
           --verbose
fi

# only n-terminus cleavage
for ntc in $ntc_grid
do
    res="dev/res-ntc-$ntc.csv"
    log="dev/res-ntc-$ntc.log"
    if [ ! -f $log ]; then
        python design_strobe_spacers.py dev/made-epitopes.csv $res \
            --max-spacer-length 4 \
            --min-spacer-length 0 \
            --num-epitopes 5 \
            --min-nterminus-cleavage $ntc \
            --log-file $log \
            --verbose
    fi
done


# only inner epitope cleavage
for epi in $epi_grid
do
    res="dev/res-epi-$epi.csv"
    log="dev/res-epi-$epi.log"
    if [ ! -f $log ]; then
        python design_strobe_spacers.py dev/made-epitopes.csv $res \
            --max-spacer-length 4 \
            --min-spacer-length 0 \
            --num-epitopes 5 \
            --max-epitope-cleavage $epi \
            --log-file $log \
            --verbose
    fi
done


# n-terminus cleavage + inner epitope cleavage, ignoring first 4
for ntc in $ntc_grid
do
    for epi in $epi_grid
    do
        res="dev/res-comb-$ntc-$epi.csv"
        log="dev/res-comb-$ntc-$epi.log"
        if [ ! -f $log ]; then
            python design_strobe_spacers.py dev/made-epitopes.csv $res \
                --max-spacer-length 4 \
                --min-spacer-length 0 \
                --num-epitopes 5 \
                --min-nterminus-cleavage $ntc \
                --max-epitope-cleavage $epi \
                --epitope-cleavage-ignore-first 4 \
                --log-file $log \
                --verbose
        fi
    done
done


# n-terminus cleavage + c-terminus cleavage + inner epitope cleavage, ignoring first 4
for ntc in $ntc_grid
do
    for epi in $epi_grid_big
    do
        res="dev/res-comb-nc-$ntc-$epi.csv"
        log="dev/res-comb-nc-$ntc-$epi.log"
        if [ ! -f $log ]; then
            python design_strobe_spacers.py dev/made-epitopes.csv $res \
                --max-spacer-length 4 \
                --min-spacer-length 0 \
                --num-epitopes 5 \
                --min-nterminus-cleavage $ntc \
                --min-cterminus-cleavage $ntc \
                --max-epitope-cleavage $epi \
                --epitope-cleavage-ignore-first 4 \
                --log-file $log \
                --verbose
        fi
    done
done


# n-terminus cleavage + c-terminus cleavage + inner epitope cleavage, ignoring first 2
for ntc in $ntc_grid
do
    for epi in $epi_grid_big
    do
        res="dev/res-comb-nc1-$ntc-$epi.csv"
        log="dev/res-comb-nc1-$ntc-$epi.log"
        if [ ! -f $log ]; then
            python design_strobe_spacers.py dev/made-epitopes.csv $res \
                --max-spacer-length 4 \
                --min-spacer-length 0 \
                --num-epitopes 5 \
                --min-nterminus-cleavage $ntc \
                --min-cterminus-cleavage $ntc \
                --max-epitope-cleavage $epi \
                --epitope-cleavage-ignore-first 2 \
                --log-file $log \
                --verbose
        fi
    done
done


# n-terminus cleavage + inner epitope cleavage + max spacer cleavage of 0, ignoring first 4
for ntc in $ntc_grid
do
    for epi in $epi_grid_big

    do
        res="dev/res-comb2-$ntc-$epi.csv"
        log="dev/res-comb2-$ntc-$epi.log"
        if [ ! -f $log ]; then
            python design_strobe_spacers.py dev/made-epitopes.csv $res \
                   --max-spacer-length 4 \
                   --min-spacer-length 0 \
                   --num-epitopes 5 \
                   --min-nterminus-cleavage $ntc \
                   --max-epitope-cleavage $epi \
                   --max-spacer-cleavage 0.0 \
                   --epitope-cleavage-ignore-first 4 \
                   --log-file $log \
                   --verbose
        fi
    done
done


# n-terminus cleavage + inner epitope cleavage, ignoring first 2
for ntc in $ntc_grid
do
    for epi in $epi_grid_big
    do
        res="dev/res-margin1-$ntc-$epi.csv"
        log="dev/res-margin1-$ntc-$epi.log"
        if [ ! -f $log ]; then
            python design_strobe_spacers.py dev/made-epitopes.csv $res \
                   --max-spacer-length 4 \
                   --min-spacer-length 0 \
                   --num-epitopes 5 \
                   --min-nterminus-cleavage $ntc \
                   --max-epitope-cleavage $epi \
                   --epitope-cleavage-ignore-first 2 \
                   --log-file $log \
                   --verbose
        fi
    done
done


# n-terminus cleavage + max spacer cleavage, with inner epitope < 0, ignoring first 2
for ntc in $ntc_grid
do
    for spa in $spa_grid
    do
        res="dev/res-spac-$ntc-$spa.csv"
        log="dev/res-spac-$ntc-$spa.log"

        if [ ! -f $log ]; then
            python design_strobe_spacers.py dev/made-epitopes.csv $res \
                --max-spacer-length 4 \
                --min-spacer-length 0 \
                --num-epitopes 5 \
                --min-nterminus-cleavage $ntc \
                --max-epitope-cleavage 0.0 \
                --epitope-cleavage-ignore-first 2 \
                --max-spacer-cleavage $spa \
                --log-file $log \
                --verbose
        fi
    done
done


# n-terminus cleavage + min spacer cleavage, with inner epitope < 0, ignoring first 2
for ntc in $ntc_grid
do
    for spa in $spa_grid
    do
        res="dev/res-spacmin-$ntc-$spa.csv"
        log="dev/res-spacmin-$ntc-$spa.log"

        if [ ! -f $log ]; then
            python design_strobe_spacers.py dev/made-epitopes.csv $res \
                --max-spacer-length 4 \
                --min-spacer-length 0 \
                --num-epitopes 5 \
                --min-nterminus-cleavage $ntc \
                --max-epitope-cleavage 0.0 \
                --epitope-cleavage-ignore-first 2 \
                --min-spacer-cleavage $spa \
                --log-file $log \
                --verbose
        fi
    done
done
