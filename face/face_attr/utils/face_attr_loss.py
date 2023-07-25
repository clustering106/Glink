import sys

from torch._C import device
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../"))

import yaml
import torch
import torch.nn as nn
import numpy as np

class BCELoss(nn.Module):
    def __init__(self, pos_weight=1, reduction='mean'):
        super(BCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        self.eps = 1e-12

    def forward(self, logits, target, mask=None):
        # logits = logits.squeeze()
        # logits: [N, *], target: [N, *]

        # logits = torch.sigmoid(logits)
        loss = -self.pos_weight * target * torch.log(logits + self.eps) - (1 - target) * torch.log(1 - logits + self.eps)
        # loss = torch.relu(logits) - logits * target + torch.log(1 + torch.exp(-torch.abs(logits)))
        if mask is not None:
            loss = loss * mask

        if self.reduction == 'mean':
            # loss = loss.mean()
            loss = loss.sum() / max(1, mask.sum())
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss

class VFLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=2, alpha=0.75):
        super(VFLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true, mask=None):
 
        loss = self.loss_fcn(pred, true)
 
        pred_prob = pred # torch.sigmoid(pred)  # prob from logits

        pos_mask = (true > 0.0).float()

        focal_weight = pos_mask * true  + (1.0 - pos_mask) * self.alpha  * torch.abs(true - pred_prob) ** self.gamma
        
        loss *= focal_weight

        if mask is not None:
            loss = loss * mask
 
        if self.reduction == 'mean':
            # loss = loss.mean()
            loss = loss.sum() / max(1, mask.sum())
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss

class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=2, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true, mask=None):
        loss = self.loss_fcn(pred, true)

        pred_prob = pred # torch.sigmoid(pred)  # prob from logits
        # alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        # loss *= alpha_factor * modulating_factor
        loss *=  modulating_factor

        if mask is not None:
            loss = loss * mask
 
        if self.reduction == 'mean':
            # loss = loss.mean()
            loss = loss.sum() / max(1, (mask > 0).sum())
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss

class WingLoss(nn.Module):
    def __init__(self, w=10, e=2, reduction='mean'):
        super(WingLoss, self).__init__()
        self.reduction = reduction
        self.w = w
        self.e = e
        self.C = self.w - self.w * np.log(1 + self.w / self.e)

    def forward(self, pred, target, mask=None):
        abs_diff = torch.abs(pred - target)
        flag = (abs_diff < self.w).float()
        loss = flag * self.w * torch.log(1 + abs_diff / self.e) + (1 - flag) * (abs_diff - self.C)

        if mask is not None:
            loss = loss * mask

        if self.reduction == 'mean':
            # loss = loss.mean()
            loss = loss.sum() / max(1, mask.sum())
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

class SmoothL1Loss(nn.Module):
    def __init__(self, beta=0.5, reduction='mean'):
        super(SmoothL1Loss, self).__init__()
        self.reduction = reduction
        self.beta = beta

    def forward(self, pred, target, mask=None):
        # pred = pred.squeeze()
        # print(pred.size(), target.size())
        # print(target)
        abs_diff = torch.abs(pred - target)
        cond = abs_diff < self.beta
        loss = torch.where(cond, 0.5 * abs_diff ** 2 / self.beta, abs_diff - 0.5 * self.beta)
        if mask is not None:
            loss = loss * mask

        if self.reduction == 'mean':
            # loss = loss.mean()
            loss = loss.sum() / max(1, (mask > 0).sum())
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
    
class MSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MSELoss, self).__init__()
        self.reduction = reduction
        self.criterion = torch.nn.MSELoss(reduction='none')

    def forward(self, pred, target, mask=None):
        loss = torch.log(self.criterion(pred, target))

        if mask is not None:
            loss = loss * mask

        if self.reduction == 'mean':
            # loss = loss.mean()
            loss = loss.sum() / max(1, (mask > 0).sum())
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

class ComputeLoss:
    # def __init__(self, model, cfg):
    #     super(ComputeLoss, self).__init__()
    #     self.device = next(model.parameters()).device  # get model device
    #     self.imgs = cfg['image_size']

    #     # self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=self.device))
    #     # self.BCEcls = nn.BCELoss(reduction='none')
    #     self.BCEcls = BCELoss()
    #     # self.BCEcls = QFocalLoss(BCELoss())
    #     self.MSEcls = MSELoss()
        
    #     self.smt_age = SmoothL1Loss(beta=3)
    #     # self.smt_age = BCELoss()

    #     self.smt_land = WingLoss(w=1.0 / self.imgs)

    #     #'pretty', 'blur', 'glass', 'makeup', 'gender', 'mouthopen', 'smile', 'eyeopen', 'isface', 'block'
    #     # score_loss, gender_loss, age_loss, land_loss, glass_loss, smile_loss, hat_loss, mask_loss
    #     # self.score_gain = cfg['score_gain']
    #     # self.gender_gain = cfg['gender_gain']
    #     # self.age_gain = cfg['age_gain']
    #     # self.land_gain = cfg['land_gain']
    #     # self.glass_gain = cfg['glass_gain']
    #     # self.smile_gain = cfg['smile_gain']
    #     # self.hat_gain = cfg['hat_gain']
    #     # self.mask_gain = cfg['mask_gain']
    #     # ['pretty', 'blur', 'glass', 'makeup', 'gender', 'mouthopen', 'smile']
    #     self.pretty_gain = cfg['pretty_gain']
    #     self.blur_gain = cfg['blur_gain']
    #     self.glass_gain = cfg['glass_gain']
    #     self.makeup_gain = cfg['makeup_gain']
    #     self.gender_gain = cfg['gender_gain']
    #     self.mouthopen_gain = cfg['mouthopen_gain']
    #     self.smile_gain = cfg['smile_gain']
    #     self.eyeopen_gain = cfg['eyeopen_gain']
    #     self.isface_gain = cfg['isface_gain']
    #     self.block_gain = cfg['block_gain']
    #     self.yaw_gain = cfg['yaw_gain']

    #     self.age_scale = torch.tensor([     10, 
    #     10,         10,         10,         10,         10,         10,         10,         10,         10,         10, 
    #     10,         10,         10,         10,        9.4,        6.3,        5.2,        3.1,        1.8,        1.4, 
    #    1.3,        1.1,        1.1,          1,          1,        1.1,        1.1,        1.2,        1.2,        1.2, 
    #    1.2,        1.3,        1.4,        1.5,        1.5,        1.5,        1.6,        1.6,        1.7,        2.1, 
    #    2.2,        2.2,        2.2,        2.3,        2.2,        2.3,        2.6,        2.6,        2.7,        2.9, 
    #    3.1,        3.3,        3.4,        3.6,        3.7,        4.4,        5.6,        6.3,        7.2,        7.1, 
    #    8.1,        9.3,         10,         10,         10,         10,         10,         10,         10,         10, 
    #     10,         10,         10,         10,         10,         10,         10,         10,         10,         10, 
    #     10,         10,         10,         10,         10,         10,         10,         10,         10,         10, 
    #     10,         10,         10,         10,         10,         10,         10,         10,         10,         10], device=self.device)

    #     # self.age_scale = torch.ones(101, device=self.device) * torch.mean(self.age_scale)
        
    # def __call__(self, preds, targets):
    #     # print(len(preds), len(targets), targets.shape)

    #     pretty_pred, blur_pred, glass_pred, makeup_pred, gender_pred, mouthopen_pred, smile_pred, eyeopen_pred, isface_pred, block_pred, yaw_pred, pitch_pred = [x.squeeze() for x in preds]
    #     # print(score_pred.shape, gender_pred.shape, age_pred.shape, land_pred.shape, glass_pred.shape, smile_pred.shape, hat_pred.shape, mask_pred.shape)
        
    #     pretty_label = targets[:, 0]
    #     blur_label = targets[:, 1]
    #     glass_label = targets[:, 2]
    #     makeup_label = targets[:, 3]
    #     gender_label = targets[:, 4]
    #     mouthopen_label = targets[:, 5]
    #     smile_label = targets[:, 6]
    #     eyeopen_label = targets[:, 7]
    #     isface_label = targets[:, 8]
    #     block_label = targets[:, 9]
    #     yaw_label = targets[:, 10]
    #     pitch_label = targets[:, 11]
    #     # print(score_label.shape, gender_label.shape, age_label.shape, land_label.shape, glass_label.shape, smile_label.shape, hat_label.shape, mask_label.shape)

    #     #['pretty', 'blur', 'glass', 'makeup', 'gender', 'mouthopen', 'smile', 'eyeopen', 'isface', 'block', 'yaw', 'pitch']
    #     pretty_mask = (pretty_label != -1)
    #     blur_mask = (blur_label != -1)
    #     glass_mask = (glass_label != -1)
    #     makeup_mask = (makeup_label != -1)
    #     gender_mask = (gender_label != -1)
    #     mouthopen_mask = (mouthopen_label != -1)
    #     smile_mask = (smile_label != -1)
    #     eyeopen_mask = (eyeopen_label != -1)
    #     isface_mask = (isface_label != -1)
    #     block_mask = (block_label != -1)
    #     yaw_mask = (yaw_label != -1)
    #     pitch_mask = (pitch_label != -1)
    #     # print(score_mask.shape, gender_mask.shape, age_mask.shape, land_mask.shape, glass_mask.shape, smile_mask.shape, hat_mask.shape, mask_mask.shape)
        
    #     pretty_loss = self.BCEcls(pretty_pred, pretty_label, pretty_mask) * self.pretty_gain
    #     blur__loss = self.BCEcls(blur_pred, blur_label, blur_mask) * self.blur_gain
    #     glass_loss = self.BCEcls(glass_pred, glass_label, glass_mask) * self.glass_gain
    #     makeup_loss = self.BCEcls(makeup_pred, makeup_label, makeup_mask) * self.makeup_gain
    #     gender_loss = self.BCEcls(gender_pred, gender_label, gender_mask) * self.gender_gain
    #     mouthopen_loss = self.BCEcls(mouthopen_pred, mouthopen_label, mouthopen_mask) * self.mouthopen_gain
    #     smile_loss = self.BCEcls(smile_pred, smile_label,  smile_mask) * self.smile_gain
    #     eyeopen_loss = self.BCEcls(eyeopen_pred, eyeopen_label, eyeopen_mask) * self.eyeopen_gain
    #     isface_loss = self.BCEcls(isface_pred, isface_label, isface_mask) * self.isface_gain
    #     block_loss = self.BCEcls(block_pred, block_label,  block_mask) * self.block_gain
    #     yaw_loss = self.smt_age(yaw_pred, yaw_label, yaw_mask) * self.yaw_gain
    #     pitch_loss = self.smt_age(pitch_pred, pitch_label, pitch_mask) * self.yaw_gain

    #     # print(score_loss, gender_loss, age_loss, land_loss, glass_loss, smile_loss, hat_loss, mask_loss)
    #     # loss = (pretty_loss + blur__loss + glass_loss + makeup_loss + gender_loss + mouthopen_loss + smile_loss + eyeopen_loss + isface_loss)
    #     # loss = (pretty_loss + blur__loss + glass_loss + makeup_loss + gender_loss + mouthopen_loss + smile_loss + eyeopen_loss + isface_loss + block_loss)
    #     # loss = (yaw_loss + pitch_loss)
    #     loss = (pretty_loss + blur__loss + glass_loss + makeup_loss + gender_loss + mouthopen_loss + smile_loss + eyeopen_loss + isface_loss + block_loss + yaw_loss + pitch_loss)
    #     return loss, torch.stack((pretty_loss, blur__loss, glass_loss, makeup_loss, gender_loss, mouthopen_loss, smile_loss, 
    #                               eyeopen_loss, isface_loss, block_loss, yaw_loss, pitch_loss)).detach()
    def __init__(self, model, cfg):
        super(ComputeLoss, self).__init__()
        self.device = next(model.parameters()).device
        self.imgs = cfg['image_size']
        self.loss_weights = {
            'pretty': cfg['pretty_gain'],
            'blur': cfg['blur_gain'],
            'glass': cfg['glass_gain'],
            'makeup': cfg['makeup_gain'],
            'gender': cfg['gender_gain'],
            'mouthopen': cfg['mouthopen_gain'],
            'smile': cfg['smile_gain'],
            'eyeopen': cfg['eyeopen_gain'],
            'isface': cfg['isface_gain'],
            'block': cfg['block_gain'],
            'yaw': cfg['yaw_gain'],
            'pitch': cfg['pitch_gain'],
            'age': cfg['age_gain'],
            'mask': cfg['mask_gain']
        }

        self.loss_functions = {
            'BCE': BCELoss(),
            'MSE': MSELoss(),
            'SmoothL1': SmoothL1Loss(beta=3),
            'Wing': WingLoss(w=1.0 / self.imgs)
        }
        
    def initialize_loss_functions(self):
        self.BCEcls = self.loss_functions['BCE']
        self.smt_age = self.loss_functions['SmoothL1']
        self.smt_land = self.loss_functions['Wing']

    def compute_loss(self, key, pred, label, maskt, loss_func, gain): 
        if key == 'age':
            # self.age_scale = torch.tensor([     10, 
            # 10,         10,         10,         10,         10,         10,         10,         10,         10,         10, 
            # 10,         10,         10,         10,        9.4,        6.3,        5.2,        3.1,        1.8,        1.4, 
            # 1.3,        1.1,        1.1,          1,          1,        1.1,        1.1,        1.2,        1.2,        1.2, 
            # 1.2,        1.3,        1.4,        1.5,        1.5,        1.5,        1.6,        1.6,        1.7,        2.1, 
            # 2.2,        2.2,        2.2,        2.3,        2.2,        2.3,        2.6,        2.6,        2.7,        2.9, 
            # 3.1,        3.3,        3.4,        3.6,        3.7,        4.4,        5.6,        6.3,        7.2,        7.1, 
            # 8.1,        9.3,         10,         10,         10,         10,         10,         10,         10,         10, 
            # 10,         10,         10,         10,         10,         10,         10,         10,         10,         10, 
            # 10,         10,         10,         10,         10,         10,         10,         10,         10,         10, 
            # 10,         10,         10,         10,         10,         10,         10,         10,         10,         10], device=self.device)
            # loss = loss_func(pred * 100, label, maskt * self.age_scale[label.long()]) * gain
            loss = loss_func(pred , label, maskt) * gain
        else:    
            loss = loss_func(pred, label, maskt) * gain
        return loss
    
    def __call__(self, preds, targets):
        self.initialize_loss_functions()
        losses = {}

        pred_labels = {
            'pretty': preds[0].squeeze(),
            'blur': preds[1].squeeze(),
            'glass': preds[2].squeeze(),
            'makeup': preds[3].squeeze(),
            'gender': preds[4].squeeze(),
            'mouthopen': preds[5].squeeze(),
            'smile': preds[6].squeeze(),
            'eyeopen': preds[7].squeeze(),
            'isface': preds[8].squeeze(),
            'block': preds[9].squeeze(),
            'yaw': preds[10].squeeze(),
            'pitch': preds[11].squeeze(),
            'age' : preds[12].squeeze(),
            'mask': preds[13].squeeze()
        }

        labels = {
            'pretty': targets[:, 0],
            'blur': targets[:, 1],
            'glass': targets[:, 2],
            'makeup': targets[:, 3],
            'gender': targets[:, 4],
            'mouthopen': targets[:, 5],
            'smile': targets[:, 6],
            'eyeopen': targets[:, 7],
            'isface': targets[:, 8],
            'block': targets[:, 9],
            'yaw': targets[:, 10],
            'pitch': targets[:, 11],
            'age' : targets[:, 12],
            'mask': targets[:, 13]
        }

        masks = {
            'pretty': (labels['pretty'] != -1),
            'blur': (labels['blur'] != -1),
            'glass': (labels['glass'] != -1),
            'makeup': (labels['makeup'] != -1),
            'gender': (labels['gender'] != -1),
            'mouthopen': (labels['mouthopen'] != -1),
            'smile': (labels['smile'] != -1),
            'eyeopen': (labels['eyeopen'] != -1),
            'isface': (labels['isface'] != -1),
            'block': (labels['block'] != -1),
            'yaw': (labels['yaw'] != -1),
            'pitch': (labels['pitch'] != -1),
            'age' : (labels['age'] != -1),
            'mask' : (labels['mask'] != -1)
        }

        for key in pred_labels:

            loss_func = self.BCEcls if key in ['pretty', 'blur', 'glass', 'makeup', 'gender', 'mouthopen', 'smile', 'eyeopen', 'isface', 'block', 'mask'] else self.smt_age
            loss = self.compute_loss(key, pred_labels[key], labels[key], masks[key], loss_func, self.loss_weights[key])
            losses[key] = loss

        total_loss = sum(losses.values())
        return total_loss, torch.stack(list(losses.values())).detach()


if __name__ == "__main__":
    config_file = "/home/zhangkai/Myproject/face_attr/configs/face_attr.yamlconfigs/face_attr.yaml"
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    
    print("end loss process !!!")