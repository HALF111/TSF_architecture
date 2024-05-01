# Experiments on exploring model architecture

cd PatchTST_supervised/
sh scripts/all_models/etth1.sh
sh scripts/all_models/etth2.sh
......

目前all_models有4个脚本，剩余的数据集的脚本可以参考Autoformer/FEDformer中的scripts来完成。


需要跑的模型目前包括自回归/非自回归的encoder-decoder（即Transformer），自回归/非自回归的Decoder，非自回归的Encoder、Masked_encoder和Fixed_decoder，共计7个。


注意1：patch_size和stride必须一样大，也即是非重叠的。

注意2：输入长度seq_len和预测长度pred_len必须都是patch_size的倍数，否则不完全的切割可能会出问题。

注意3：为了保证参数量一致，各个模型间需要保证d_model等超参数一致；此外统一将Encoder-only或Decoder-only中的e_layers或d_layers设置为6；而在Encoder-Decoder中，e_layers和d_layers分别设置为3。
