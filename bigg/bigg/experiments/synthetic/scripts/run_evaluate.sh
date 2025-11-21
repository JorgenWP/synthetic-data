


prediction_file=../../../../results/lobster/DFS-blksize--1-b-80/epoch-25.ckpt.graphs-0
test_file=../../../../data/lobster-DFS/test-graphs.pkl
metrics=degree,clustering,spectral

python evaluate.py \
  -pred_file $prediction_file \
  -test_file $test_file \
  -metrics $metrics \