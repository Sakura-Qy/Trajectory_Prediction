import argparse
import pickle
import os
import torch
from utils.tools import attrs5
import pandas as pd
import numpy as np
import random


from exp.exp_informer import Exp_Informer


parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

parser.add_argument('--model', type=str, required=False, default='informer',
                    help='model of experiment, options: [informer, informerstack, informerlight(TBD), caral, carag, carpag, carpaginformer, carppag]')

# parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
parser.add_argument('--data', type=str, required=False, default='laneC3', help='data')

parser.add_argument('--root_path', type=str, default='./data', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='laneC.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='s_Location.1', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='s',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

parser.add_argument('--seq_len', type=int, default=30,
                    help='input sequence length of Informer encoder')  # 输入长度 3s [96,36,48]
parser.add_argument('--label_len', type=int, default=24,
                    help='start token length of Informer decoder')  # 解码器开始位置 [72,48,48] [84,48,48]
parser.add_argument('--pred_len', type=int, default=12 * 4, help='prediction sequence length')  # 预测长度 3s
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

parser.add_argument('--enc_in', type=int, default=64, help='encoder input size')  # 和输入数据维度有关
parser.add_argument('--dec_in', type=int, default=2, help='decoder input size')
parser.add_argument('--c_out', type=int, default=2, help='output size')

parser.add_argument('--d_model', type=int, default=512, help='dimension of model')  # xian 1/2
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=1024, help='dimension of fcn')  # xian 1/2
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.15, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]') # fixed
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data', default=True)
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=15, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=True)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

args = parser.parse_args()


args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

Exp = Exp_Informer

file = {
    'laneC4': ['./laneC4.pkl', './sample_list4.pkl', attrs5],  # 采样12hz,修改d_Velocity
}
args.data = 'laneC4'

# 修改输入模型的序列长度以及模型的预测长度
args.seq_len = 60           # 输入模型长度 以24帧记录
args.label_len = 48         # 指导向量长度
args.pred_len = 24 * 5      # 预测轨迹时长

# 修改进行还是预测还是训练
args.do_predict = True
args.do_predict = False

# 预测的车辆序号
args.car_index = 301

# 采用12Hz
args.seq_len = int(args.seq_len / 2)
args.label_len = int(args.label_len / 2)
args.pred_len = int(args.pred_len / 2)

pkl_file, sample_file, attrs = file[args.data]

args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ', '').split(',')]
args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)

# 提取数据
# 其中lanec4保存了所有车辆所有时刻的数据
# sample_list4保存了每一辆车包含的数据长度，目的为方便dataloader控制放入模型的数据
if os.path.exists(pkl_file):
    print('Existing pkl file, open it get better performance')
    with open(pkl_file, 'rb') as f:
        df_raw = pickle.load(f)
        df_raw = df_raw[attrs]
    with open(sample_file, 'rb') as f:  # 测试数据集
        samples_list = pickle.load(f)
else:
    raise ValueError('不存在数据')

# 选择训练的车辆数量以及用于预测的车辆，确保没有交集
if args.do_predict:
    a, b = 10000, 11000
else:
    a, b = 0, 3000

list_l = sum(samples_list[:a])
list_r = sum(samples_list[:b])
samples_list = samples_list[a:b]  # len 12301

df_raw = df_raw.iloc[list_l:list_r]

args.enc_in = len(df_raw.columns) - 1
print('enc_in is ', args.enc_in)

# 开始运行
for ii in range(args.itr):
    # setting record of experiments
    setting = f'{float(args.seq_len / 12)}s_{int(args.pred_len / 12)}s_xy_12hz_loss_cancel_time_exp_{args.model}'

    exp = Exp(args, df_raw, samples_list)  # set experiments

    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)
    else:
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

    torch.cuda.empty_cache()
