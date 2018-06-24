#!/bin/bash
filename="$1"

sed -E  '/^Prob.*$/d' $filename > $filename.rmprob


paste -d ' ' dev.eval $filename.rmprob > $filename.rmprob.3col


./conlleval < $filename.rmprob.3col 
score=`./conlleval < $filename.rmprob.3col | sed -n 2p`
mv $filename.rmprob.3col $filename.rmprob.3col.${score: -5}


