python design.py dev/made-epitopes.csv dev/res-fixed-spacer-aay.csv \
    -S 3 -s 3  -e 5 \
    --min-cterminus-cleavage 1.6 \
    --min-nterminus-cleavage 1.6 \
    --max-epitope-cleavage 1.0  \
    --epitope-cleavage-ignore-first 3 \
    --fix-spacer AAY \
    --verbose

python design.py dev/made-epitopes.csv dev/res-fixed-spacer-mwqw.csv \
    -S 4 -s 4  -e 5 \
    --min-cterminus-cleavage 1.95 \
    --min-nterminus-cleavage 1.95 \
    --max-epitope-cleavage 0.0  \
    --epitope-cleavage-ignore-first 3 \
    --fix-spacer MWQW \
    --verbose
