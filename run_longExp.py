import argparse
import os
import torch
from exp.exp_main import Exp_Main
# from exp.exp_main_new_head import Exp_Main_new_head
import random
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Informer, Transformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


    # DLinear
    #parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

    # PatchTST
    # 本文一共有3个dropout，分别为：dropout、fc_dropout、head_dropout
    # fc_dropout是指最后一层如果是预训练层（会用一维卷积的预测头），对应的dropout
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    # head_dropout是指最后一层如果是"展平+线性映射层"，对应的dropout
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    # 默认是需要做padding的！
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    # 默认是做RevIN的！！但RevIN的affine参数并不学习
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    # RevIN默认减去均值（也可以像NLinear一样减掉最后一个值）
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    # 默认不做分解
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    # 默认也是不使用独立的头！
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # Formers 
    # 编码方式
    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    # Informer中对Q进行采样时，对Q采样的因子数
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    # 这里是全局的dropout，例如encoder前做embedding时的dropout
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    # 时间特征嵌入方法
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    # parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    # 这里是否默认设置为0会更好？因为加载数据并不是瓶颈，多线程反倒可能由于别人在跑程序导占用较多的cpu而导致变慢
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    # 学习率修改策略
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    # 使用混合精度进行训练
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    
    # * 加入一个norm策略，用于判断batchnorm还是layernorm
    parser.add_argument('--norm', type=str, default='batch', help='batch, layer')
    
    # L2正则
    parser.add_argument('--add_l2', action='store_true')
    parser.add_argument('--l2_alpha', type=float, default=1e-4)
    
    # 在预测头中额外引入的d_pred参数（optional）
    parser.add_argument('--d_pred', type=int, default=512, help='intermediate layer dimension in the prediction head')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
    
    parser.add_argument('--run_train', action='store_true')
    parser.add_argument('--run_test', action='store_true')
    parser.add_argument('--get_attn_plot', action='store_true')
    
    # 获得ACF-forecastability矩阵
    parser.add_argument('--get_forecastability', action="store_true")
    
    # 增加针对当前数据集的最长的seq_len
    parser.add_argument('--longest_seq_len', type=int, default=2000)
    
    # 新增只取出部分训练集数据
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--multiple_pred_len_list', nargs='+')
    parser.add_argument('--run_multiple_pred_len', action='store_true')

    args = parser.parse_args()

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)


    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    # 具体用哪个可以根据真实情况来修改
    Exp = Exp_Main
    # Exp = Exp_Main_new_head

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            # setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            #     args.model_id,
            #     args.model,
            #     args.data,
            #     args.features,
            #     args.seq_len,
            #     args.label_len,
            #     args.pred_len,
            #     args.d_model,
            #     args.n_heads,
            #     args.e_layers,
            #     args.d_layers,
            #     args.d_ff,
            #     args.factor,
            #     args.embed,
            #     args.distil,
            #     args.des,ii)
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_fore{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.get_forecastability,
                args.des,
                ii)

            exp = Exp(args)  # set experiments
            
            # if args.get_forecastability:
            #     print('>>>>>>>get_forecastability : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting)) 
            #     exp.get_forecastability(setting)
            
            if args.run_train:
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)

            if args.run_test:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting, test=1)
                
            if args.run_multiple_pred_len:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test_multiple_pred_len(setting, test=1)
            
            if args.get_attn_plot:
                print('>>>>>>>get attn_plot : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                
                # assert args.batch_size == 1
                
                # def grid_show(to_shows, cols):
                #     rows = (len(to_shows)-1) // cols + 1
                #     it = iter(to_shows)
                #     fig, axs = plt.subplots(rows, cols, figsize=(rows*8.5, cols*2))
                #     for i in range(rows):
                #         for j in range(cols):
                #             try:
                #                 image, title = next(it)
                #             except StopIteration:
                #                 image = np.zeros_like(to_shows[0][0])
                #                 title = 'pad'
                #             axs[i, j].imshow(image)
                #             axs[i, j].set_title(title)
                #             axs[i, j].set_yticks([])
                #             axs[i, j].set_xticks([])
                #     print("Saving figures in grid_show...")
                #     folder = "./attn_weights"
                #     plt.savefig(f"{folder}/{args.model_id}.pdf")
                #     plt.show()

                def visualize_head(att_map, name):
                    plt.figure()
                    ax = plt.gca()
                    # Plot the heatmap
                    im = ax.imshow(att_map)
                    # Create colorbar
                    cbar = ax.figure.colorbar(im, ax=ax)
                    
                    print("Saving figures in visualize_head...")
                    folder = f"./attn_weights/{args.model_id}"
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    plt.savefig(f"{folder}/{name}.pdf")
                    plt.show()
                    
                # def visualize_heads(att_map, cols):
                #     to_shows = []
                #     att_map = att_map.squeeze()
                #     for i in range(att_map.shape[0]):
                #         to_shows.append((att_map[i], f'Head {i}'))
                #     average_att_map = att_map.mean(axis=0)
                #     to_shows.append((average_att_map, 'Head Average'))
                #     grid_show(to_shows, cols=cols)

                def gray2rgb(image):
                    return np.repeat(image[...,np.newaxis],3,2)
                
                cache = exp.attn_plot(setting)
                print(type(cache))
                print(list(cache.keys()))
                
                # 事实上，假设batch_size为B，总数据量为(S/B)个batch，数据channel为C，注意力头数为A，
                # 每个样本中token个数为N，以及encoder层数为L
                # 那么结果为(S/B)*L个数组；
                # 其中每个数组包含每个样本在每个层的注意力值，也即[C, A, N, N]
                attention_maps = cache["TSTEncoderLayer.forward"]
                
                print(len(attention_maps))  # =(S/B)*L个
                sample_num = len(attention_maps) // args.e_layers
                
                # a = 0
                # for i in range(sample_num):
                #     if i % 1000 == 0:
                #         for l in range(args.e_layers):
                #             # for a in range(args.n_heads):
                #                 for c in range(args.enc_in):
                #                     visualize_head(attention_maps[i*args.e_layers + l][c, a], f"s{i}_l{l}_head{a}_c{c}")
                #                     # visualize_head(attention_maps[i*args.e_layers + l][c, 0], f"s{i}_l{l}_head{a}_c{c}")
                
                # 事实上，我们在这里可以做一些均值的操作
                # stack后为[(S/B)*L*C, A, N, N]
                attention_maps_stack = np.vstack(attention_maps)
                print(attention_maps_stack.shape)
                # 然后如果对全部avg的话那么只剩[N, N]了
                attention_maps_avg = np.mean(attention_maps_stack, axis=(0,1))
                print(attention_maps_avg.shape)
                
                visualize_head(attention_maps_avg, "avg_all")
                

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                    args.model,
                                                                                                    args.data,
                                                                                                    args.features,
                                                                                                    args.seq_len,
                                                                                                    args.label_len,
                                                                                                    args.pred_len,
                                                                                                    args.d_model,
                                                                                                    args.n_heads,
                                                                                                    args.e_layers,
                                                                                                    args.d_layers,
                                                                                                    args.d_ff,
                                                                                                    args.factor,
                                                                                                    args.embed,
                                                                                                    args.distil,
                                                                                                    args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
        