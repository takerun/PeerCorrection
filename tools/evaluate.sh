#!/bin/sh

read TEST
for var in PC1 PC2 PG1 PG3 PG4 PG5 PCG1 PCG3 PCG4 PCG5
do
    python evaluatePCG.py -m $var -tune cor -test $TEST
done
