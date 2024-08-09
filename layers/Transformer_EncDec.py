import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        # 注意池化层的步长为2
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))  # 交换后两维，但是由于kernel=3、而padding=1，所以维度的数值没有变化
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)  # 由于max_pool的stride=2，所以maxpool的步长为2，这会导致x中的原来的长度96会减少一半变成48了
        x = x.transpose(1, 2)  # 后两维再交换回来
        
        # 假设输入为[B, L, D]，那么输出由于做了池化，会变成[B, L/2, D]
        return x


# TODO:增加将LN变成BN的code！！

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", norm_type="layer"):
        super(EncoderLayer, self).__init__()
        
        # d_ff为Encoder中的postional-wise FFN中，两个全连接层里面的隐藏层维度，默认为4*d_model
        d_ff = d_ff or 4 * d_model
        
        # 传入的attention一般是一个AttentionLayer类的实例
        self.attention = attention
        
        # 从d_model到d_ff的一维卷积，后面再卷回来
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # Transformer一般用LayerNorm
        if norm_type == "layer":
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        elif norm_type == "batch":
            # * 也可以改用BatchNorm！
            from layers.PatchTST_layers import Transpose
            self.norm1 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            self.norm2 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))

        # 从大的结构来看，Informer和Transformer中的Encoder部分的结构基本是类似的；
        # 主要区别就是attention计算方法做了替换，其他的残差连接是类似的
        # 也即 attetion -> add&norm -> positional-wise FFN -> add&norm

        # 这里就相当于在做自注意力，所以QKV三者均传入了x
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        # 然后先dropout，再add（残差连接），再norm；也即第一个add & norm
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        
        # 然后下面也是做和transformer一样的positional-wise FFN，
        # 这其中包括两个全连接层和中间一个ReLU激活层，且全连接层的中间隐藏层维度为d_ff
        # 不过这里用了kernel=1的一维卷积、来替代全连接层了，但本质上二者是一样的，都是一个全连接/MLP
        # PS：new_x、x和y的维度都是[32, 96, 512]，也即[B, L, D]
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        # 这里相当于是做第二个add & norm
        # 返回输出结果x和注意力attn
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            # 将attn_layers和conv_layers两两结对，形成许多个pair，并会遍历这些pair。长度按照较短的那个来计算
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                # 如果有conv，那么一层注意力、一层卷积交替进行
                x, attn = attn_layer(x, attn_mask=attn_mask)  # 经过attn_layer后，x的维度不变，仍然为[B,L,D]，也即[32, 96, 512]（但是随着层数向上，96这一维每过一层就会除2；会不断变成48，24，12……）
                x = conv_layer(x)  # conv_layer就是为了蒸馏操作：也即在Informer的Encoder层中，每层的维度是越往上越小的，每过一层都会除2
                attns.append(attn)
            
            # 由于一般attn_layer都要比conv_layer多一层（因为attn_layer层有e_layers个，而conv_layer层只有e_layers-1个），所以还要再用attn_layer[-1]额外做一次
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                # 没有conv的话，就只需要做注意力（里面已包含了FFN）即可了
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        # 最后再做一次norm，这里一般是要使用LayerNorm，而不是BatchNorm
        # PS：但是PatchTST里面用的就是BatchNorm1d(d_model)？？
        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu", norm_type="layer"):
        super(DecoderLayer, self).__init__()
        
        d_ff = d_ff or 4 * d_model
        
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        
        # 同样是从d_model到d_ff的一维卷积，后面再卷回来
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # Transformer一般用LayerNorm
        # 这里由于有3个子层，所以做3次norm
        if norm_type == "layer":
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
        elif norm_type == "batch":
            from layers.PatchTST_layers import Transpose
            self.norm1 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            self.norm2 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            self.norm3 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # 1. 第一层，先做decoder自己的自注意力
        # 注意：这里一般是带mask的自注意力！！！
        # 以及别忘记每次add&norm前都要先dropout以下，下面同理
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        # 2. 第二层，是做正常的attention，其中queries来自decoder，而keys和values则来自encoder
        # 但是这里的x和cross的维度是不一样的，x为[32, 72, 512]，而cross则为[32, 48, 512]，所以维度相关信息需要在FullAttention中完成处理
        # 上述操作是没问题的，因为这里的72是query的个数n，48为key-value对的个数m，这二者本来就可以不相同的
        # 也就是Transformer里有两个保证，假设Q为[B, n, d_q]，K为[B, m, d_k]，Q为[B, m, d_v]
        # 要求一个是K和V的个数m必须相同；另一个是d_q和d_k必须相同。
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        # 第二次的add & norm
        y = x = self.norm2(x)
        
        # 3. 最后也要做一次 positional-wise FFN（这和Encoder中的操作基本一致）
        # 一维卷积类似于MLP？
        # 以及第三次的add & norm
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # self.layers中一共会含有d_layers层的DecoderLayer层
        # 其中每个DecoderLayer层又会包括两个attention层，第一层为self-attention层，第二层为正常的attention层
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        # 这里再做一次norm，一般是LayerNorm
        # * 但这里可能是BatchNorm！
        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x



# 由于Decoder-only模型中不包含cross-attention，所以这里也要做个修改！
class DecoderLayer_wo_CrossAttn(nn.Module):
    def __init__(self, self_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu", norm_type="layer"):
        super(DecoderLayer_wo_CrossAttn, self).__init__()
        
        d_ff = d_ff or 4 * d_model
        
        self.self_attention = self_attention
        
        self.norm_type = norm_type
        
        # 同样是从d_model到d_ff的一维卷积，后面再卷回来
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # Transformer一般用LayerNorm
        # 这里由于有2个子层，所以做2次norm
        if norm_type == "layer":
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        elif norm_type == "batch":
            from layers.PatchTST_layers import Transpose
            self.norm1 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            # self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            # self.norm2 = nn.BatchNorm1d(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, x_mask=None, cross_mask=None):
        # 1. 第一层，先做decoder自己的自注意力
        # 注意：这里一般是带mask的自注意力！！！
        # 以及别忘记每次add&norm前都要先dropout以下，下面同理
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)
        
        # 不做cross-attention了，所以这里保留一个y。
        y = x
        
        # 3. 最后也要做一次 positional-wise FFN（这和Encoder中的操作基本一致）
        # 一维卷积类似于MLP？
        # 以及第三次的add & norm
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


class Decoder_wo_CrossAttn(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder_wo_CrossAttn, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, x_mask=None, cross_mask=None):
        # self.layers中一共会含有d_layers层的DecoderLayer层
        # 其中每个DecoderLayer层又会包括两个attention层，第一层为self-attention层，第二层为正常的attention层
        for layer in self.layers:
            x = layer(x, x_mask=x_mask, cross_mask=cross_mask)

        # 这里再做一次norm，一般是LayerNorm
        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
    

class Decoder_wo_CrossAttn_wo_proj(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder_wo_CrossAttn_wo_proj, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        # self.projection = projection

    def forward(self, x, x_mask=None, cross_mask=None):
        # self.layers中一共会含有d_layers层的DecoderLayer层
        # 其中每个DecoderLayer层又会包括两个attention层，第一层为self-attention层，第二层为正常的attention层
        for layer in self.layers:
            x = layer(x, x_mask=x_mask, cross_mask=cross_mask)

        # 这里再做一次norm，一般是LayerNorm
        if self.norm is not None:
            x = self.norm(x)

        # if self.projection is not None:
        #     x = self.projection(x)
        return x



class EncoderLayer_w_CrossAttn(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, 
                 dropout=0.1, activation="relu", norm_type="layer"):
        super(EncoderLayer_w_CrossAttn, self).__init__()
        
        # d_ff为Encoder中的postional-wise FFN中，两个全连接层里面的隐藏层维度，默认为4*d_model
        d_ff = d_ff or 4 * d_model
        
        # 传入的attention一般是一个AttentionLayer类的实例
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        
        # 从d_model到d_ff的一维卷积，后面再卷回来
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # Transformer一般用LayerNorm
        if norm_type == "layer":
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
        elif norm_type == "batch":
            # * 也可以改用BatchNorm！
            from layers.PatchTST_layers import Transpose
            self.norm1 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            self.norm2 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            self.norm3 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # x [B, L, D]
        # 这里就相当于在做自注意力，所以QKV三者均传入了x
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)
        
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        # 第二次的add & norm
        y = x = self.norm2(x)
        
        # 然后下面也是做和transformer一样的positional-wise FFN，
        # 这其中包括两个全连接层和中间一个ReLU激活层，且全连接层的中间隐藏层维度为d_ff
        # 不过这里用了kernel=1的一维卷积、来替代全连接层了，但本质上二者是一样的，都是一个全连接/MLP
        # PS：new_x、x和y的维度都是[32, 96, 512]，也即[B, L, D]
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        # 这里相当于是做第二个add & norm
        # 返回输出结果x和注意力attn
        return self.norm3(x + y)

class Encoder_w_CrossAttn_w_proj(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder_w_CrossAttn_w_proj, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # self.layers中一共会含有d_layers层的DecoderLayer层
        # 其中每个DecoderLayer层又会包括两个attention层，第一层为self-attention层，第二层为正常的attention层
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        # 这里再做一次norm，一般是LayerNorm
        # * 但这里可能是BatchNorm！
        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        
        return x