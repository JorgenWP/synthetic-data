


g_type=transactions
ordering=BFS
start_date=2018-01-01
cutoff_date=2019-01-01

save_dir=../../data/Transactions/$g_type-$ordering

if [ ! -e $save_dir ]; then
  mkdir -p $save_dir
fi

python preprocess_transactions.py \
  -save_dir $save_dir \
  -node_order $ordering \
  -start_date $start_date \
  -cutoff_date $cutoff_date \