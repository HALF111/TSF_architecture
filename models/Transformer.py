import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding,DataEmbedding_wo_pos,DataEmbedding_wo_temp,DataEmbedding_wo_pos_temp
import numpy as np
from layers.RevIN import RevIN

from layers.PatchTST_layers import Transpose

class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        
        # * 注意，PatchTST参考TST的设计，其中默认使用的是BatchNorm
        # * 而不像In/Auto/FED-former等默认使用LayerNorm
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len  # 这里context_window就等于seq_len
        target_window = configs.pred_len  # target_window就等于pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        # patch_len = configs.patch_len
        # stride = configs.stride
        # padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        
        # 新加一个norm
        norm = configs.norm
        
        # RevIn
        # 这里默认是做RevIN的，且维度c_in为channel维
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Embedding
        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        elif configs.embed_type == 1:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        elif configs.embed_type == 3:
            self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 4:
            self.enc_embedding = DataEmbedding_wo_pos_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        
        if "batch" in norm.lower():
            # ! 为什么batchnorm要先转置再做呢？
            # PS：对于3D的输入数据，BatchNorm1d的参数必须和数据第二维度的数值相同
            # 其输入为：Input: (N, C)或者(N, C, L)，其中N是batch大小，C是channel数量，L是序列长度
            # 所以这里是要把d_model放到中间，seq_len或pre_len放在后面
            self.norm_layer = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        else:
            # * layernorm的输入是(N,...)就可以了，LN对于当前batch内数据的所有维度做归一化
            self.norm_layer = nn.LayerNorm(configs.d_model)
        
        # Encoder
        # 其中包含e_layers个EncoderLayer层，和e_layers-1个ConvLayer层
        # 其中EncoderLayer中还需要传入一个AttentionLayer层
        # 最后再加上一个LayerNorm层
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,  # 这里为relu或gelu
                    norm_type=configs.norm,
                ) for l in range(configs.e_layers)
            ],
            # norm_layer=torch.nn.LayerNorm(configs.d_model)
            norm_layer=self.norm_layer
        )
        
        # Decoder
        # Decoder中包含d_layers个DecoderLayer，
        # 其中每个DecoderLayer中包括一个使用ProbAttention的自注意力层，和一个使用FullAttention的正常的注意力层（这个注意力层的queries来自上一层自注意力层，而keys和values则来自encoder）
        # 最后也还是再加上一层layerNorm层
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),  # 注意这里第一个的mask_flag参数被设置成为了True，这是因为在decoder的第一层用的是masked的自注意力；而其他的注意力都是False
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    norm_type=configs.norm,
                )
                for l in range(configs.d_layers)
            ],
            # norm_layer=torch.nn.LayerNorm(configs.d_model),
            norm_layer=self.norm_layer,
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        # 假设batch_size=32, seq_len=96, label_len=24, pred_len=48, channel=12, mark_channel=4
        # x_enc为[32, 96, 12], x_mark_enc为[32, 96, 4], x_dec为[32, 72, 12], x_mark_dec为[32, 72, 4]
        # 也即输入维度为[batch_size, seq_len, channel]
        
        # print(self.norm_layer)
        
        # norm
        # 1、先做RevIN归一化
        # ref：https://github.com/ts-kim/RevIN/blob/master/baselines/Informer2020/models/model.py
        if self.revin: 
            x_enc = self.revin_layer(x_enc, 'norm')  # 沿着seq_len维度做归一化？

        # 先对encoder的输入数据做一次embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [bs, sl, d_model]
        # 将embedding后的数据输入到encoder中
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)  # [bs, sl, d_model]

        # 对于decoder，我们同样需要做一次embedding
        dec_out = self.dec_embedding(x_dec, x_mark_dec)  # [bs, pl, d_model]
        # 之后embedding后的数据和上面encoder的结果共同送入decoder中
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)  # [bs, pl, c]
        
        if self.revin:
            dec_out = self.revin_layer(dec_out, 'denorm')

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
