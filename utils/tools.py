import os
import numpy as np
import torch
from torch import nn


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj=='type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif args.lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        names = os.listdir(path)
        if names:
            for each in names:
                bench_mark = each.split('_')[0]
                try:
                    if float(bench_mark) > val_loss and float(bench_mark) > 0.32:
                        file = os.path.join(path, each)
                        os.remove(file)
                except:
                    pass
        torch.save(model.state_dict(), path+'/'+str(val_loss)[:6]+'_checkpoint.pth')
        self.val_loss_min = val_loss

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)
        # np.savez('Standardscaler_all.npz', mean=self.mean, std=self.std)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data, pred=False):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if pred:            # 
            mean = mean[:-2]
            std = std[:-2]
        elif data.shape[-1] != mean.shape[-1]:
            mean = mean[-2:]
            std = std[-2:]

        return (data * std) + mean

class CoordinateMSELoss(nn.Module):
    def __init__(self):
        super(CoordinateMSELoss, self).__init__()

    def forward(self, true, pred):
        squared_diff = torch.pow(true - pred, 2)
        loss = torch.mean(squared_diff.view(squared_diff.size(0), -1).sum(dim=1))
        loss = torch.sqrt(loss)

        return loss

attrs5 = ['date', 'scene_avg_speed', 'current_lane_average_speed', 'distance_of_left_lane', 'distance_of_right_lane',
          'vehicle_width', 'vehicle_height', 's_Location', 'd_Location', 's_Velocity', 'd_Velocity', 's_Acceleration',
          'd_Acceleration', 'heading_angle', 'left_lane_type', 'right_lane_type',
          'yaw_rate', 's_jerk', 'd_jerk', 'ttc', 'thw', 'dhw', 'steering_rate_entropy',
          'vehicle_pre_current_s_location', 'vehicle_pre_current_d_location',
          'vehicle_pre_current_s_velocity', 'vehicle_pre_current_d_velocity', 'vehicle_pre_width', 'vehicle_pre_height',
          'vehicle_pre_gap',
          'vehicle_follow_current_s_location', 'vehicle_follow_current_d_location', 'vehicle_follow_current_s_velocity',
          'vehicle_follow_current_d_velocity', 'vehicle_follow_width',
          'vehicle_follow_height', 'vehicle_follow_gap', 'vehicle_left_pre_current_s_location',
          'vehicle_left_pre_current_d_location', 'vehicle_left_pre_current_s_velocity',
          'vehicle_left_pre_current_d_velocity', 'vehicle_left_pre_width', 'vehicle_left_pre_height',
          'vehicle_left_pre_gap', 'vehicle_left_follow_current_s_location', 'vehicle_left_follow_current_d_location',
          'vehicle_left_follow_current_s_velocity', 'vehicle_left_follow_current_d_velocity',
          'vehicle_left_follow_width', 'vehicle_left_follow_height', 'vehicle_left_follow_gap',
          'vehicle_right_pre_current_s_location', 'vehicle_right_pre_current_d_location',
          'vehicle_right_pre_current_s_velocity', 'vehicle_right_pre_current_d_velocity', 'vehicle_right_pre_width',
          'vehicle_right_pre_height', 'vehicle_right_pre_gap', 'vehicle_right_follow_current_s_location',
          'vehicle_right_follow_current_d_location', 'vehicle_right_follow_current_s_velocity',
          'vehicle_right_follow_current_d_velocity', 'vehicle_right_follow_width', 'vehicle_right_follow_height',
          'vehicle_right_follow_gap', 's_diff', 'd_diff']
