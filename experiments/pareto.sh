#!/bin/bash
set -ex

# parameters: cleavage_prior upper_bound initial_epitopes
function run_if_new {
    basename="dev/res-paretonoub-$1"
    if [ ! -f "$basename.log" ]; then
        python pareto.py dev/made-epitopes.csv "$basename-{:.2f}.csv" \
               --verbose --log-file "$basename.log" \
               --num-epitopes 5 \
               --mc-draws 100 \
               --cleavage-prior $1 \
               --rounds 20 \
               --upper-bound $2
    fi
}


run_if_new 0.1 0.8 "-se NILLHPMSL -se SILLHPMSL -se YTPGPGPRF -se YTPGPGPRY -se YYKNHIRTL"
run_if_new 0.15 0.75 "-se SILLHPMSL -se MAREIHPEY -se AAMDISHFL -se YTPGPGPRF -se YYKNHIRTL"
run_if_new 0.2 0.5 "-se SILLHPMSL -se AAMDISHFL -se MAREIHPEY -se GAFDFCFFF -se YYKNHIRTL"
run_if_new 0.25 0.5 "-se YYKNHIRTL -se SILLHPMSL -se GAFDFCFFF -se AAMDISHFL -se MAREIHPEY"
run_if_new 0.3 0.5 "-se SKFCGWPAV -se AAMDISHFL -se MAREIHPEY -se YTPGPGPRF -se GAFDFCFFF"
