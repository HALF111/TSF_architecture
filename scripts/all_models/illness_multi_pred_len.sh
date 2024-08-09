root_path_name=./dataset/
data_path_name=national_illness.csv
model_id_name=national_illness
data_name=custom

# seq_len=104
# model_name=PatchTST
# model_name=Transformer
# model_name=Transformer_patch

gpu_num=0

random_seed=2021


# for model_name in PatchTST Encoder Encoder_overall Encoder_zeros Masked_encoder Prefix_decoder Decoder Transformer Encoder_zeros_flatten Masked_encoder_flatten Double_decoder Double_encoder
# for model_name in Transformer
for model_name in Encoder Encoder_overall Encoder_zeros_no_flatten Masked_encoder_no_flatten Prefix_decoder Decoder Transformer
do
if [[ "$model_name" =~ "Transformer" || "$model_name" =~ "Double" ]]; then
    e_layers=3
    d_layers=3
elif [[ "$model_name" =~ "Encoder" || "$model_name" =~ "encoder" || "$model_name" =~ "PatchTST" ]]; then
    e_layers=6
    d_layers=0
elif [[ "$model_name" =~ "Decoder" || "$model_name" =~ "decoder" || "$model_name" =~ "PatchTST" ]]; then
    e_layers=0
    d_layers=6
fi
# for norm in layer batch
for norm in layer
do
for seq_len in 120
do
# ! 注意：pred_len必须是最大的那个！
# for pred_len in 24
for pred_len in 60
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
      --train_epochs 100\
      --patch_len 6 \
      --stride 6 \
      --gpu $gpu_num \
      --batch_size 32 \
      --norm $norm \
      --multiple_pred_len_list 24 36 48 60 \
      --run_train --run_multiple_pred_len \
      > './script_outputs/'$model_id_name'_'$seq_len'_'$pred_len'/'$model_name'_'multiple_pred_len.log
done
done
done
done