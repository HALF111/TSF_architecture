root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

# seq_len=104
# model_name=PatchTST
model_name=Transformer
# model_name=Transformer_patch

gpu_num=3

random_seed=2021
# for pred_len in 24 36 48 60
for seq_len in 336
do
for pred_len in 96
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --d_model 512 \
      --des 'Exp' \
      --itr 1 \
      --train_epochs 100\
      --gpu $gpu_num \
      --batch_size 32 \
      --run_train --run_test \
      --norm batch
done
done