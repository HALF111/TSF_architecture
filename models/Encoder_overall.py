import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding,DataEmbedding_wo_pos,DataEmbedding_wo_temp,DataEmbedding_wo_pos_temp
import numpy as np
from layers.RevIN import RevIN

class Model(nn.Module):
    """
    Transformer with patching + channel independence
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
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        
        # 记录一些self的变量：
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # RevIn
        # 这里默认是做RevIN的，且维度c_in为channel维
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Patching
        # PS：这里context_window就等于seq_len，target_window就等于pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        # 获得总的patch数
        self.input_patch_num = int((context_window - patch_len) / stride + 1)
        self.output_patch_num = int((target_window - patch_len) / stride + 1)
        
        # ! 是不是要保证非重叠的切割？
        assert self.stride == self.patch_len
        assert self.input_patch_num * self.patch_len == context_window
        assert self.output_patch_num * self.patch_len == target_window
        
        # if padding_patch == 'end':  # can be modified to general case
        #     # (0, stride)表示左边的padding为0个，右边的padding为stride个
        #     self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
        #     self.input_patch_num += 1
        #     self.output_patch_num += 1

        # Embedding
        if configs.embed_type == 0:
            # self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
            #                                 configs.dropout)
            # self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
            #                                configs.dropout)
            
            # ! 这里的embedding应当改为从patch_len映射到d_model上！
            self.enc_embedding = DataEmbedding(self.patch_len, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout)
            self.dec_embedding = DataEmbedding(self.patch_len, configs.d_model, configs.embed, configs.freq,
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
        
        from layers.PatchTST_layers import Transpose
        # 新加一个norm
        norm = configs.norm
        if "batch" in norm.lower():
            # PS：对于3D的输入数据，BatchNorm1d的参数必须和数据第二维度的数值相同
            # 其输入为：Input: (N, C)或者(N, C, L)，其中N是batch大小，C是channel数量，L是序列长度
            # 而这里的输入是(batch_size, patch_num, d_model)，所以需要转置下将d_model放到前面去。
            # 所以这里是要把d_model放到中间，patch_num放在后面
            self.norm_layer = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        else:
            self.norm_layer = nn.LayerNorm(configs.d_model)
            
        # * mask embedding
        # * 这个就是masked encoder中加在预测窗口中的embedding
        with torch.no_grad():
            self.mask = torch.zeros(1, d_model)  # (1, d_model)
        
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
        
        self.flatten = nn.Flatten(start_dim=-2)
        self.total_patch_num = self.input_patch_num + self.output_patch_num
        self.projection = nn.Linear(self.total_patch_num*configs.d_model, self.pred_len)
        
        # # Decoder
        # # Decoder中包含d_layers个DecoderLayer，
        # # 其中每个DecoderLayer中包括一个使用ProbAttention的自注意力层，和一个使用FullAttention的正常的注意力层（这个注意力层的queries来自上一层自注意力层，而keys和values则来自encoder）
        # # 最后也还是再加上一层layerNorm层
        # self.decoder = Decoder(
        #     [
        #         DecoderLayer(
        #             AttentionLayer(
        #                 FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
        #                 configs.d_model, configs.n_heads),  # 注意这里第一个的mask_flag参数被设置成为了True，这是因为在decoder的第一层用的是masked的自注意力；而其他的注意力都是False
        #             AttentionLayer(
        #                 FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
        #                 configs.d_model, configs.n_heads),
        #             configs.d_model,
        #             configs.d_ff,
        #             dropout=configs.dropout,
        #             activation=configs.activation,
        #             norm_type=configs.norm,
        #         )
        #         for l in range(configs.d_layers)
        #     ],
        #     # norm_layer=torch.nn.LayerNorm(configs.d_model),
        #     norm_layer=self.norm_layer,
        #     # projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        #     # ! 注意这里的projection的映射维度也要改成patch_len
        #     projection=nn.Linear(configs.d_model, self.patch_len, bias=True)
        # )

    def forward(self, x_enc, x_mark_enc,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        # 假设batch_size=32, seq_len=96, label_len=24, pred_len=48, channel=12, mark_channel=4
        # x_enc为[32, 96, 12], x_mark_enc为[32, 96, 4], x_dec为[32, 72, 12], x_mark_dec为[32, 72, 4]
        # 也即输入维度为[batch_size, seq_len, channel]
        
        # norm
        # 1、先做RevIN归一化
        # ref：https://github.com/ts-kim/RevIN/blob/master/baselines/Informer2020/models/model.py
        if self.revin: 
            x_enc = self.revin_layer(x_enc, 'norm')  # 沿着seq_len维度做归一化？
            # x_dec = self.revin_layer(x_dec, 'norm')  # x_dec都是0，感觉不需要做norm
            
        # do patching
        # # 2、做patching分割
        # # 先做padding
        # if self.padding_patch == 'end':
        #     x_enc = self.padding_patch_layer(x_enc)
        #     x_dec = self.padding_patch_layer(x_dec)
            
        # unfold函数啊按照选定的尺寸与步长来切分矩阵，相当于滑动窗口操作，也即只有卷、没有积
        # 参数为（dim,size,step）：dim表明想要切分的维度，size表明切分块（滑动窗口）的大小，step表明切分的步长
        # 这里用于从一个分批输入的张量中提取滑动的局部块
        x_enc = x_enc.permute(0,2,1)                                                # [batch_size, channel, seq_len]
        x_enc = x_enc.unfold(dimension=-1, size=self.patch_len, step=self.stride)   # [bs x channel x patch_num x patch_len]
        # z = z.permute(0,1,3,2)  # [bs x nvars x patch_len x patch_num]
        bs, nvars, input_patch_num, patch_len = x_enc.shape
        x_enc = x_enc.reshape(bs*nvars, input_patch_num, patch_len)  # [(bs*channel) x patch_num x patch_len]
        
        # # * x_dec同理
        # x_dec = x_dec.permute(0,2,1)                                                # [batch_size, channel, seq_len]
        # x_dec = x_dec.unfold(dimension=-1, size=self.patch_len, step=self.stride)   # [bs x channel x patch_num x patch_len]
        # bs, nvars, output_patch_num, patch_len = x_dec.shape
        # x_dec = x_dec.reshape(bs*nvars, output_patch_num, patch_len)  # [(bs*channel) x patch_num x patch_len]


        # ! 先生成mask_token
        # 2、加上预测窗口中的mask部分
        # self.mask: [1, d_model]
        mask_token = self.mask.unsqueeze(0)  # mask_token: [1, 1, d_model]
        mask_token = mask_token.repeat(x_enc.shape[0], self.output_patch_num, 1)  # mask_token: [bs x nvars x output_patch_num x d_model]
        mask_token = mask_token.to(x_enc.device)

        # 先对encoder的输入数据做一次embedding
        # ! 注意，这里不需要x_mark_enc信息了！其中的位置编码足以标志出信息！
        enc_out = self.enc_embedding(x_enc, mask_token=mask_token)  # [(bs*channel) x total_patch_num x d_model]
        # 将embedding后的数据输入到encoder中
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)  # [(bs*channel) x total_patch_num x d_model]
        
        # # ! 由于mask_encoder包含前面部分，这里需要取出最后的output_patch_num的部分
        # # ! 但是由于是overall，所以包含整个enc_out都没问题
        # enc_out = enc_out[:, -self.output_patch_num:, :]  # [(bs*channel) x output_patch_num x d_model]
        
        # * 先展平，然后做线性映射？
        output = self.flatten(enc_out)  # [(bs*channel) x output_patch_num x patch_len]
        output = self.projection(output)  # [(bs*channel) x pred_len]
        output = output.reshape(bs, nvars, -1)  # [bs x channel x pred_len]
        output = output.permute(0,2,1)  # [bs x pred_len x channel]
        
        if self.revin:
            # print(dec_out.shape)
            # print(self.revin_layer.mean.shape)
            # print(self.revin_layer.stdev.shape)
            output = self.revin_layer(output, 'denorm')  # [bs x pred_len x channel]

        if self.output_attention:
            return output[:, -self.pred_len:, :], attns
        else:
            return output[:, -self.pred_len:, :]  # [B, L, D]
