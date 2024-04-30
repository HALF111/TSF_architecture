gpu_num=1

dir_name=all_result

# for model in FEDformer Autoformer Informer
for model in Autoformer
do
# for pred_len in 96 192
for pred_len in 96
do

# # ETTm1
# name=ETTm1
# cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
# if [ ! -d $cur_path ]; then
#     mkdir $cur_path
# fi
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm1.csv \
#   --task_id ETTm1 \
#   --model $model \
#   --data ETTm1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len $pred_len \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 512 \
#   --itr 1 \
#   --gpu $gpu_num \
#   --run_train --run_test \
#   > $cur_path'/'train_and_test_loss.log

# ETTh1
name=ETTh1
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
# cur_path=./$dir_name/$model'_'$name'_'pl$pred_len'_'revin
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --task_id ETTh1 \
  --model $model \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --itr 1 \
  --gpu $gpu_num \
  --run_train --run_test \
  > $cur_path'/'train_and_test_loss.log

# # ETTm2
# name=ETTm2
# cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
# if [ ! -d $cur_path ]; then
#     mkdir $cur_path
# fi
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm2.csv \
#   --task_id ETTm2 \
#   --model $model \
#   --data ETTm2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len $pred_len \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 512 \
#   --itr 1 \
#   --gpu $gpu_num \
#   --run_train --run_test \
#   > $cur_path'/'train_and_test_loss.log

# # ETTh2
# name=ETTh2
# cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
# if [ ! -d $cur_path ]; then
#     mkdir $cur_path
# fi
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh2.csv \
#   --task_id ETTh2 \
#   --model $model \
#   --data ETTh2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len $pred_len \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 512 \
#   --itr 1 \
#   --gpu $gpu_num \
#   --run_train --run_test \
#   > $cur_path'/'train_and_test_loss.log

# # electricity
# name=ECL
# cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
# if [ ! -d $cur_path ]; then
#     mkdir $cur_path
# fi
# python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/electricity/ \
#  --data_path electricity.csv \
#  --task_id ECL \
#  --model $model \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len $pred_len \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 321 \
#  --dec_in 321 \
#  --c_out 321 \
#  --des 'Exp' \
#  --itr 1 \
#   --gpu $gpu_num \
#   --run_train --run_test \
#   > $cur_path'/'train_and_test_loss.log

# # exchange
# name=Exchange
# cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
# if [ ! -d $cur_path ]; then
#     mkdir $cur_path
# fi
# python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/exchange_rate/ \
#  --data_path exchange_rate.csv \
#  --task_id Exchange \
#  --model $model \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len $pred_len \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 8 \
#  --dec_in 8 \
#  --c_out 8 \
#  --des 'Exp' \
#  --itr 1 \
#   --gpu $gpu_num \
#   --run_train --run_test \
#   > $cur_path'/'train_and_test_loss.log

# # traffic
# name=Traffic
# cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
# if [ ! -d $cur_path ]; then
#     mkdir $cur_path
# fi
# python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/traffic/ \
#  --data_path traffic.csv \
#  --task_id traffic \
#  --model $model \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len $pred_len \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 862 \
#  --dec_in 862 \
#  --c_out 862 \
#  --des 'Exp' \
#  --itr 1 \
#  --train_epochs 10 \
#   --gpu $gpu_num \
#   --run_train --run_test \
#   > $cur_path'/'train_and_test_loss.log

# # weather
# name=Weather
# cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
# if [ ! -d $cur_path ]; then
#     mkdir $cur_path
# fi
# python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/weather/ \
#  --data_path weather.csv \
#  --task_id weather \
#  --model $model \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len $pred_len \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 21 \
#  --dec_in 21 \
#  --c_out 21 \
#  --des 'Exp' \
#  --itr 1 \
#   --gpu $gpu_num \
#   --run_train --run_test \
#   > $cur_path'/'train_and_test_loss.log
done

# for pred_len in 24 36 48 60
# do
# # illness
# name=Illness
# cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
# if [ ! -d $cur_path ]; then
#     mkdir $cur_path
# fi
# python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/illness/ \
#  --data_path national_illness.csv \
#  --task_id ili \
#  --model $model \
#  --data custom \
#  --features M \
#  --seq_len 36 \
#  --label_len 18 \
#  --pred_len $pred_len \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --itr 1 \
#   --gpu $gpu_num \
#   --run_train --run_test \
#   > $cur_path'/'train_and_test_loss.log
# done

done