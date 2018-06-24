#!/bin/bash
filename="$1"

sed -E  '/^Prob.*$/d' $filename > $filename.rmprob


paste -d ' ' test.eval $filename.rmprob > $filename.rmprob.2col

cat $filename.rmprob.2col | sed  -e 's/^[ \t]*//' > $filename.rmprob.2col.submit

rm $filename.rmprob.2col



