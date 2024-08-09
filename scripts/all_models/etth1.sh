root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

# seq_len=104
# model_name=PatchTST
# model_name=Transformer
# model_name=Transformer_patch

gpu_num=0

random_seed=2021


# ! 注意：需要用"bash etth1.sh"调用而非"sh etth1.sh"来调用此script
# for model_name in Encoder Encoder_overall Encoder_zeros Masked_encoder Prefix_decoder Decoder Transformer
for model_name in Encoder Encoder_overall Encoder_zeros_flatten Encoder_zeros_no_flatten Masked_encoder_flatten Masked_encoder_no_flatten Prefix_decoder Decoder Transformer Double_encoder Double_decoder
do
if [[ "$model_name" =~ "Encoder" || "$model_name" =~ "encoder" ]]; then
    e_layers=6
    d_layers=0
elif [[ "$model_name" =~ "Decoder" || "$model_name" =~ "decoder" || "$model_name" =~ "PatchTST" ]]; then
    e_layers=0
    d_layers=6
elif [[ "$model_name" =~ "Transformer" ]]; then
    e_layers=3
    d_layers=3
fi
# for norm in layer batch
for norm in layer
do
for seq_len in 512
do
# for pred_len in 96
for pred_len in 96 192 336 720
do
    if [ ! -d './script_outputs/' ]; then
        mkdir './script_outputs/'
    fi
    if [ ! -d './script_outputs/'$model_id_name'_'$seq_len'_'$pred_len'/' ]; then
        mkdir './script_outputs/'$model_id_name'_'$seq_len'_'$pred_len'/'
    fi
    
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
      --e_layers $e_layers \
      --d_layers $d_layers \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --d_model 512 \
      --des 'Exp' \
      --itr 1 \
      --train_epochs 20 \
      --patch_len 16 \
      --stride 16 \
      --gpu $gpu_num \
      --batch_size 32 \
      --run_train --run_test \
      --norm $norm \
      > './script_outputs/'$model_id_name'_'$seq_len'_'$pred_len'/'$model_name'_'$norm'norm'.log
done
done
done
done