import pandas as pd

from torch.utils.data import Dataset
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings

warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    def __init__(self, root_path,  df_raw, sample_list, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None, car_index = None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4  # 24 为
            self.label_len = 24 * 4
            self.pred_len = 24 * 4  # 4+4 为模型的encoder输出
        else:
            self.seq_len = size[0]  # 为通过多少时间的数据预测 encoder:size[0]， decoder:size[1] + size[2]
            self.label_len = size[1]  # 设置pre label的长度，是已知的
            self.pred_len = size[2]  # 真实预测的长度
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.sample_queue = sample_list
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__(df_raw)

    def __read_data__(self, df_raw):
        self.scaler = StandardScaler()
        # df_raw = pd.read_csv(os.path.join(self.root_path,
        #                                   self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns); 
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
            cols_y = self.target
        else:   #走这
            cols = list(df_raw.columns)
            # cols_y = ['s_Location', 'd_Location']
            cols_y = ['s_diff', 'd_diff'] # ['s_Location', 'd_Location']
            cols_x = list(df_raw.columns)
            cols_x.remove('date')

        sample_train = int(len(self.sample_queue) * 0.85)       # 0.985     0.85
        sample_test = int(len(self.sample_queue) * 0.1)         # 0.01      0.1
        # sample_vali = len(self.sample_queue) - sample_train - sample_vali

        num_train = sum(self.sample_queue[0:sample_train])
        num_test = sum(self.sample_queue[-sample_test:])

        # data:     |train|vali|test|
        # sample:   |train|vali|test|
        # 获取sample_list，用于生成sample迭代器
        sample_vali = len(self.sample_queue) - sample_train - sample_test
        border_sl = [0, sample_train, sample_train + sample_vali]
        border_sr = [sample_train, sample_train + sample_vali, len(self.sample_queue)]

        self.sample_list = self.sample_queue[border_sl[self.set_type]:border_sr[self.set_type]]

        # 获取数据
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train, num_train + num_vali]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS': # here
            df_data = df_raw[cols_x + cols_y]
        else:  # self.features == 'S'
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2, :-2]
        if self.inverse:  # 应该选true
            self.data_y = df_data.values[border1:border2, -2:]
        else:
            self.data_y = data[border1:border2, -2:]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_y[r_begin:r_end]
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, df_raw, sample_list, flag='pred', size=None, features='S', data_path='ETTh1.csv', target='OT', scale=True,
                 inverse=False, timeenc=0, freq='s', cols=None, car_index = 100):
        # size [seq_len, label_len, pred_len]
        # info
        self.sample_queue = sample_list
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.car_index = car_index
        self.__read_data__(df_raw)

    def __read_data__(self, df_raw):
        self.scaler = StandardScaler()
        # df_raw = pd.read_csv(os.path.join(self.root_path,
        #                                   self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
            cols_y = self.target
        else:
            cols = list(df_raw.columns)
            # cols_y = ['s_Location', 'd_Location']
            cols_y = ['s_diff', 'd_diff']
            cols_x = list(df_raw.columns)
            cols_x.remove('date')
        # car_insed = -100 # 选择第几辆车作为预测
        car_insed = self.car_index
        border1 = sum(self.sample_queue[:car_insed])
        border2 = sum(self.sample_queue[:car_insed + 1])
        self.sample_list = [self.sample_queue[car_insed]]

        if self.features == 'M' or self.features == 'MS':
            df_data = df_raw[cols_x + cols_y]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2, :-2]
        if self.inverse:  # 应该选true
            self.data_y = df_data.values[border1:border2, -2:]
        else:
            self.data_y = data[border1:border2, -2:]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_begin + self.label_len + self.pred_len]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark


    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
