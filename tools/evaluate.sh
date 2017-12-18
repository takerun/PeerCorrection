#!/bin/sh

echo "Set tune metric:"
read TUNE
echo "Set test metric:"
read TEST
echo "Set name of saving file:"
read FILE
for var in PC1 PC2 PG1 PG3 PG4 PG5 PCG1 PCG3 PCG4 PCG5
do
    python evaluatePCG.py -m $var -tune $TUNE -test $TEST
done > ../result/inferred_parameter/$FILE
