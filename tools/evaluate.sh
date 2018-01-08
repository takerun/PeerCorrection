#!/bin/sh

echo "Set tune metric:"
read TUNE
echo "Set test metric:"
read TEST
echo "Set top-k value:"
read TOPK
echo "Set threshold:"
read THRESHOLD
#echo "Set name of saving file:"
#read FILE
case $THRESHOLD in
  1 ) Fthreshold="1234" ;;
  2 ) Fthreshold="234" ;;
  3 ) Fthreshold="34" ;;
  4 ) Fthreshold="4" ;;
esac

case $TUNE in
  "cor" ) Ftune="Corrcoef" ;;
  "ktau" ) Ftune="Kendalltau" ;;
  "srho" ) Ftune="Spearmanrho" ;;
  "auc" ) Ftune="AUC_${THRESHOLD}" ;;
esac

case $TEST in
  "cor" ) Ftest="corrcoef" ;;
  "ktau" ) Ftest="kendalltau" ;;
  "srho" ) Ftest="spearmanrho" ;;
  "preck" ) Ftest="precisionAt${TOPK}_${Fthreshold}" ;;
  "auc" ) Ftest="auc_${Fthreshold}" ;;
  "ndcg" ) Ftest="nDCGAt${TOPK}" ;;
esac

if [ $TUNE = $TEST ]; then
  FILE="${Ftest}_5fold.txt"
else
  FILE="${Ftest}_5fold_tuned${Ftune}.txt"
fi

# evaluate simple methods
python calculateMeanGrade.py -test $TEST -k $TOPK -thre $THRESHOLD > ../result/inferred_parameter/evaluations/$FILE
# evaluate bayesian models
for var in PC1 PC2 PC2a PG1 PG3 PG4 PG5 PCG1 PCG3 PCG4 PCG5 PG1PC2 PG3PC2 PG4PC2 PG5PC2 PG1PC2a PG3PC2a PG4PC2a PG5PC2a
do
    python evaluatePCG.py -m $var -tune $TUNE -test $TEST -k $TOPK -thre $THRESHOLD
done >> ../result/inferred_parameter/evaluations/$FILE
