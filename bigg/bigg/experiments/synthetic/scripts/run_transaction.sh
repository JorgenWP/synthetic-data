g_type=transactions
ordering=BFS
blksize=-1
bsize=5

data_dir=../../../../data/Transactions/$g_type-$ordering

save_dir=../../../../results/$g_type/$ordering-blksize-$blksize-b-$bsize

if [ ! -e $save_dir ];
then
  mkdir -p $save_dir
fi

#export CUDA_VISIBLE_DEVICES=3

python ../batch_train.py \
  -data_dir $data_dir \
  -pos_enc True \
  -tree_pos_enc False \
  -share_param True \
  -save_dir $save_dir \
  -g_type $g_type \
  -node_order $ordering \
  -num_graphs $num_g \
  -blksize $blksize \
  -epoch_save 10 \
  -bits_compress 256 \
  -batch_size $bsize \
  -num_test_gen 20 \
  -num_epochs 50 \
  -gpu 0 \
  $@