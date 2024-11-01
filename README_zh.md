# Experiments on exploring model architecture

cd TSF_architecture/
bash scripts/all_models/etth1.sh
bash scripts/all_models/etth2.sh
......

目前all_models有8个脚本，分别对应于8个数据集。其中每个脚本里包含7个模型。

注意0：这里必须用bash跑，而不是用sh跑，否则可能会报错！！

注意1：patch_size和stride必须一样大，也即是非重叠的。

注意2：输入长度seq_len和预测长度pred_len必须都是patch_size的倍数，否则不完全的切割可能会出问题。

注意3：为了保证参数量一致，其一是各个模型间需要保证d_model等超参数一致；其二是统一将Encoder-only或Decoder-only中的e_layers或d_layers设置为6；而在Encoder-Decoder中，e_layers和d_layers分别设置为3，以保证层数一致。
