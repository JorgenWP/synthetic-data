


prediction_file=../../../../results/Transactions/epoch-1000.ckpt.graphs-0.0
test_file=../../../../data/Transactions/test-graphs.pkl
metrics=degree,clustering,spectral

python preprocess_transactions.py \
  -pred_file $prediction_file \
  -test_file $test_file \
  -metrics $metrics \