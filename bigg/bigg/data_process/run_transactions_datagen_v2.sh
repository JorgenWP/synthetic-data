


input_path=../../data/Transactions/raw/GT.gpickle
graph_size=1000
sampling_method=forest_fire

g_type=transactions
ordering=BFS

save_dir=../../data/Transactions/$g_type-$ordering

if [ ! -e $save_dir ]; then
  mkdir -p $save_dir
fi

python preprocess_transactions_v2.py \
  -input_path $input_path \
  -save_dir $save_dir \
  -node_order $ordering \
  -sampling_method $sampling_method \
  -num_graphs 50 \
  -min_nodes 500 \
  -max_nodes 1000 \
  