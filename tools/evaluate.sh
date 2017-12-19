#!/bin/sh

echo "Set tune metric:"
read TUNE
echo "Set test metric:"
read TEST
echo "Set top-k value:"
read TOPK
echo "Set threshold:"
read THRESHOLD
echo "Set name of saving file:"
read FILE
for var in PC1 PC2 PG1 PG3 PG4 PG5 PCG1 PCG3 PCG4 PCG5
do
    python evaluatePCG.py -m $var -tune $TUNE -test $TEST -k $TOPK -thre $THRESHOLD
done > ../result/inferred_parameter/$FILE
