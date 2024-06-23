import json

import httpx

from data.data_loader import Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack, CarAttentionLstm, CarAttentionGru, CarProbAttentionGru, CarPAGpdecoder, CarEmbedProbAttentionGru
from utils.tools import EarlyStopping, adjust_learning_rate, CoordinateMSELoss
from utils.metrics import metric
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

import os
import time

import warnings

warnings.filterwarnings('ignore')

class MySampler(Sampler):  # 自定义sampler类
    def __init__(self, mlist, length, shuffle=True) -> None:  # mlist 是不同文件的数据长度，length是每一个batch的长度
        super(MySampler).__init__()
        self.s = mlist
        self.sampler_list = self.create_sampler(length)
        if shuffle:
            np.random.shuffle(self.sampler_list)

    def create_sampler(self, length):
        st = []
        cid = 0
        for val in self.s:
            if val > length:
                st.extend(range(cid, cid + val - length + 1))
            cid += val
        return st

    def __iter__(self):
        return iter(self.sampler_list)

    def __len__(self):
        return len(self.sampler_list)


class Exp_Informer(Exp_Basic):
    def __init__(self, args, df_raw, sample_list):
        super(Exp_Informer, self).__init__(args)
        self.in_length = int(args.seq_len) + int(args.pred_len)
        self.df_raw = df_raw
        self.sample_list = sample_list
        # self.writer = SummaryWriter('./train_board')

    def _build_model(self):
        model_dict = {
            'informer': Informer,
            'informerstack': InformerStack,
            'caral': CarAttentionLstm,      # attenton + lstm
            'carag': CarAttentionGru,       # attention + gru
            'carpag': CarProbAttentionGru,  # prob attention + gru
            'carpaginformer': CarPAGpdecoder, # pag+informer
            'carppag': CarEmbedProbAttentionGru # 添加了position embedding
        }
        e_layers = self.args.e_layers if self.args.model == 'informer' else self.args.s_layers
        model = model_dict[self.args.model](
            self.args.enc_in,
            self.args.dec_in,
            self.args.c_out,
            self.args.seq_len,
            self.args.label_len,
            self.args.pred_len,
            self.args.factor,
            self.args.d_model,
            self.args.n_heads,
            e_layers,  # self.args.e_layers,
            self.args.d_layers,
            self.args.d_ff,
            self.args.dropout,
            self.args.attn,
            self.args.embed,
            self.args.freq,
            self.args.activation,
            self.args.output_attention,
            self.args.distil,
            self.args.mix,
            self.device
        ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args
        df_raw = self.df_raw
        sample_list = self.sample_list

        # data_dict = {
        #     'laneC': Dataset_Custom,
        # }
        # Data = data_dict[self.args.data]
        Data = Dataset_Custom
        timeenc = 0 if args.embed != 'timeF' else 1

        if flag == 'test':
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            df_raw=df_raw,
            sample_list=sample_list,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols,
            car_index=args.car_index
        )

        # 添加获取sample生成原始文件，并生成sample列表

        our_sampler = MySampler(data_set.sample_list, self.in_length, shuffle_flag)  # 自定义sampler
        print(flag, len(our_sampler))

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            sampler=our_sampler,
            # shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = CoordinateMSELoss()
        # criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, epoch):
        self.model.eval()
        train_steps = len(vali_loader)
        tp = 50
        time_now = time.time()
        iter_count = 0

        with torch.no_grad():
            total_loss = []
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                iter_count += 1
                pred, true = self._process_one_batch(
                    vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                # 将差值转换为实际的x，y

                res = vali_data.scaler.inverse_transform(batch_x, True)
                last_loc = res[:, self.args.seq_len - 1, 6:8].to(self.device)  # find last location
                true[:, 0, :] += last_loc
                pred[:, 0, :] += last_loc
                for ii in range(1, true.shape[1]):
                    true[:, ii, :] += true[:, ii - 1, :]
                    pred[:, ii, :] += pred[:, ii - 1, :]
                # adjustment y
                # true[:, :, 1] *= 20
                # pred[:, :, 1] *= 20

                loss = criterion(pred.detach().cpu(), true.detach().cpu())
                total_loss.append(loss.item())
                # print(f'\r{("-" * ((i + 1) % tp) + ">"):<{tp}} | {(i + 1) % tp}/{tp}', end='')

                if (i + 1) % tp == 0:
                    print("\r\titers: {0}/{1}, epoch: {2} | loss: {3:.7f}      ".format(i + 1, train_steps, epoch,
                                                                                        loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
        show = False
        if show:
            import matplotlib.pyplot as plt

            data1 = range(len(total_loss))
            data2 = total_loss
            # Add labels and title
            plt.title("Vali Loss Plot")
            plt.xlabel("Iter")
            plt.ylabel("Loss")
            # Plot a line graph
            plt.plot(data1, data2)
            plt.show()
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            tp = 50  # 进度条长度
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)

                # 将差值转换为实际的x，y
                res = train_data.scaler.inverse_transform(batch_x, True)
                last_loc = res[:, self.args.seq_len - 1, 6:8].to(self.device)  # find last location
                true[:, 0, :] += last_loc
                pred[:, 0, :] += last_loc
                for ii in range(1, true.shape[1]):
                    true[:, ii, :] += true[:, ii - 1, :]
                    pred[:, ii, :] += pred[:, ii - 1, :]

                loss = criterion(pred, true)
                train_loss.append(loss.item())
                # print(f'\r{("-" * ((i + 1) % tp) + ">"):<{tp}} |{(i + 1) % tp}/{tp}', end='')
                if (i + 1) % tp == 0:
                    print('\r<', '-' * 56, '>')
                    print("\r\titers: {0}/{1}, epoch: {2} | loss: {3:.7f}      ".format(i + 1, train_steps, epoch + 1,
                                                                                        loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            # gc.collect()
            # torch.cuda.empty_cache()
            print("\nEpoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion, epoch + 1)
            test_loss = self.vali(test_data, test_loader, criterion, epoch + 1)

            send_data = "\nEpoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss)
            print(send_data)
            # self.writer.add_scalar('loss', vali_loss, epoch)
            # self.send_text(send_data)
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
        file_list = os.listdir(path)
        file_list = sorted(file_list, key=lambda file: os.path.getctime(os.path.join(path, file)), reverse=True)
        best_model_path = os.path.join(path, file_list[0])
        # best_model_path = path + '/' + 'mse0.58_checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()

        preds = []
        trues = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            # 将差值转换为实际的x，y
            res = test_data.scaler.inverse_transform(batch_x, True)
            last_loc = res[:, self.args.seq_len - 1, 6:8].to(self.device)  # find last location
            true[:, 0, :] += last_loc
            pred[:, 0, :] += last_loc
            for ii in range(1, true.shape[1]):
                true[:, ii, :] += true[:, ii - 1, :]
                pred[:, ii, :] += pred[:, ii - 1, :]

            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = r'C:\Users\Danny\Desktop\lane_change\Informer2020_version_2\results\ + setting + \\'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return mse, mae

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        model_file = 'none'

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            file_list = os.listdir(path)
            file_list = sorted(file_list, key=lambda file: os.path.getctime(os.path.join(path, file)), reverse=True)
            if len(file_list) > 1:
                print('Choose a model for prediction:')
                for index, each in enumerate(file_list):
                    print(f'\t{index}  {each}')
                f_index = int(input(f'Input 0-{len(file_list)-1} :'))
            else:
                f_index = 0
            model_file = file_list[f_index]
            best_model_path = os.path.join(path, model_file)
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        # torch.save(self.model, '2.5s_5s_xy_12hz.pth')
        # raise EOFError
        criterion = self._select_criterion()
        lens = len(pred_loader) + pred_data.pred_len + pred_data.seq_len - 1
        preds = []
        trues = []
        true_trajectory = pred_data.scaler.inverse_transform(pred_data.data_x, True)
        record = np.zeros([4, lens])
        record[2, :] = true_trajectory[:, 6]
        record[3, :] = true_trajectory[:, 7]
        plt.rcParams['figure.figsize'] = (12.8, 6)
        plt.show()
        self.client = httpx.Client()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)

            print(f'<-----------step:{i}----{model_file}------>')

            last_loc = torch.tensor(record[2:, pred_data.seq_len - 1 + i]).to(self.device)  # find last location
            pred[:, 0, :] += last_loc
            true[:, 0, :] += last_loc
            for j in range(1, pred.shape[1]):  # add last time dx,dy
                pred[:, j, :] += pred[:, j - 1, :]
                true[:, j, :] += true[:, j - 1, :]

            loss = criterion(pred, true)
            pred = pred.detach().cpu().numpy()
            true = true.detach().cpu().numpy()

            print(f'Loss:{loss.item()}')
            re_l = pred_data.seq_len + i
            re_r = re_l + pred_data.pred_len
            record[0, re_l:re_r] = pred[:, :, 0]
            record[1, re_l:re_r] = pred[:, :, 1]

            # ------------------record-------------------#
            # 0 pred_x
            # 1 pred_y
            # 2 history_x
            # 3 history_y
            # -------------------------------------------#

            # pred trajectory
            pred_x, pred_y = record[0, re_l:re_r], record[1, re_l:re_r]
            # true trajectory
            true_x, true_y = record[2, re_l:re_r], record[3, re_l:re_r]
            # history trajectory
            put_in_x, put_in_y = record[2, :i], record[3, :i]
            # history input trajectory
            put_x, put_y = record[2, i:re_l], record[3, i:re_l]

            plt.clf()
            if not i:
                y_min = min(put_y) - 5
                y_max = min(put_y) + 5
                x_min = min(put_x)
                x_max = x_min + 500
            plt.ylim(y_min, y_max)
            plt.xlim(x_min, x_max)
            input_t, = plt.plot(put_x, put_y)  # input trajectory
            pred_t, = plt.plot(pred_x, pred_y)  # pred trajectory
            true_t, = plt.plot(true_x, true_y)  # true trajectory
            history, = plt.plot(put_in_x, put_in_y)  # history trajectory
            plt.xlabel('S_location')
            plt.ylabel('D_location')
            plt.legend(handles=[input_t, pred_t, true_t, history],
                       labels=["input trajectory", "pred trajectory", "true trajectory", "history trajectory"],
                       loc="upper right", fontsize=10)
            plt.pause(0.1)
            # plt.show()
            preds.append(pred)
            trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        self.client.close()

        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding == 0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()  #

        elif self.args.padding == 1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()

        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)  #

        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs) # 不走

        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        return outputs, batch_y

    def _api_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, url='http://127.0.0.1:8000/predict2/'):
        data = {
            "batch_x": batch_x.tolist(),
            "batch_x_mark": batch_x_mark.tolist(),
            "batch_y": batch_y.tolist(),
            "batch_y_mark": batch_y_mark.tolist()
        }
        response = self.client.post(url, json=data, timeout=None)
        a = json.loads(response.text)
        outputs = a['prediction']
        outputs = torch.tensor(outputs)
        outputs = dataset_object.inverse_transform(outputs)
        outputs = outputs.to(self.device)
        batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
        return outputs, batch_y
