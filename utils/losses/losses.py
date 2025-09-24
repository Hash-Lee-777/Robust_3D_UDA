import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from torch.distributions import normal

class CELoss(nn.Module):
    def __init__(self, ignore_label: int = None, weight: np.ndarray = None):
        '''
        :param ignore_label: label to ignore
        :param weight: possible weights for weighted CE Loss
        '''
        super().__init__()
        if weight is not None:
            weight = torch.from_numpy(weight).float()
            print(f'----->Using weighted CE Loss weights: {weight}')

        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_label, weight=weight)
        self.ignored_label = ignore_label

    def forward(self, preds: torch.Tensor, gt: torch.Tensor):

        loss = self.loss(preds, gt)
        return loss


class DICELoss(nn.Module):

    def __init__(self, ignore_label=None, powerize=True, use_tmask=True):
        super(DICELoss, self).__init__()

        if ignore_label is not None:
            self.ignore_label = torch.tensor(ignore_label)
        else:
            self.ignore_label = ignore_label

        self.powerize = powerize
        self.use_tmask = use_tmask

    def forward(self, output, target):
        input_device = output.device
        # temporal solution to avoid nan
        output = output.cpu()
        target = target.cpu()

        if self.ignore_label is not None:
            valid_idx = torch.logical_not(target == self.ignore_label)
            target = target[valid_idx]
            output = output[valid_idx, :]

        target = F.one_hot(target, num_classes=output.shape[1])
        output = F.softmax(output, dim=-1)

        intersection = (output * target).sum(dim=0)
        if self.powerize:
            union = (output.pow(2).sum(dim=0) + target.sum(dim=0)) + 1e-12
        else:
            union = (output.sum(dim=0) + target.sum(dim=0)) + 1e-12
        if self.use_tmask:
            tmask = (target.sum(dim=0) > 0).int()
        else:
            tmask = torch.ones(target.shape[1]).int()

        iou = (tmask * 2 * intersection / union).sum(dim=0) / (tmask.sum(dim=0) + 1e-12)

        dice_loss = 1 - iou.mean()

        return dice_loss.to(input_device)


def get_soft(t_vector, eps=0.25):

    max_val = 1 - eps
    min_val = eps / (t_vector.shape[-1] - 1)

    t_soft = torch.empty(t_vector.shape)
    t_soft[t_vector == 0] = min_val
    t_soft[t_vector == 1] = max_val

    return t_soft


def get_kitti_soft(t_vector, labels, eps=0.25):

    max_val = 1 - eps
    min_val = eps / (t_vector.shape[-1] - 1)

    t_soft = torch.empty(t_vector.shape)
    t_soft[t_vector == 0] = min_val
    t_soft[t_vector == 1] = max_val

    searched_idx = torch.logical_or(labels == 6, labels == 1)
    if searched_idx.sum() > 0:
        t_soft[searched_idx, 1] = max_val/2
        t_soft[searched_idx, 6] = max_val/2

    return t_soft

class SoftDICELoss(nn.Module):
    def __init__(self, ignore_label=None, powerize=True, use_tmask=True,
                 neg_range=False, eps=0.05, is_kitti=False):
        super(SoftDICELoss, self).__init__()

        if ignore_label is not None:
            self.ignore_label = torch.tensor(ignore_label)
        else:
            self.ignore_label = ignore_label
        self.powerize = powerize
        self.use_tmask = use_tmask
        self.neg_range = neg_range
        self.eps = eps
        self.is_kitti = is_kitti

    def forward(self, output, target, return_class=False, is_kitti=False):
        input_device = output.device
        # temporal solution to avoid nan
        output = output.cpu()
        target = target.cpu()

        # Filter out -1 labels
        valid_idx = target != -1
        target = target[valid_idx]
        output = output[valid_idx, :]

        if self.ignore_label is not None:
            valid_idx = torch.logical_not(target == self.ignore_label)
            target = target[valid_idx]
            output = output[valid_idx, :]
        target_onehot = F.one_hot(target, num_classes=output.shape[1])
        if not self.is_kitti and not is_kitti:
            target_soft = get_soft(target_onehot, eps=self.eps)
        else:
            target_soft = get_kitti_soft(target_onehot, target, eps=self.eps)

        output = F.softmax(output, dim=-1)

        intersection = (output * target_soft).sum(dim=0)

        if self.powerize:
            union = (output.pow(2).sum(dim=0) + target_soft.sum(dim=0)) + 1e-12
        else:
            union = (output.sum(dim=0) + target_soft.sum(dim=0)) + 1e-12
        if self.use_tmask:
            tmask = (target_onehot.sum(dim=0) > 0).int()
        else:
            tmask = torch.ones(target_onehot.shape[1]).int()

        iou = (tmask * 2 * intersection / union).sum(dim=0) / (tmask.sum(dim=0) + 1e-12)
        iou_class = tmask * 2 * intersection / union

        if self.neg_range:
            dice_loss = -iou.mean()
            dice_class = -iou_class
        else:
            dice_loss = 1 - iou.mean()
            dice_class = 1 - iou_class
        if return_class:
            return dice_loss.to(input_device), dice_class
        else:
            return dice_loss.to(input_device)

# KPSloss
class KPSLoss(nn.Module):
    def __init__(self, cls_num_list=None, max_m=0.5, weighted=True, weight=None, s=30, is_kitti=False):
        super(KPSLoss, self).__init__()
        assert s > 0

        s_list = torch.cuda.FloatTensor(cls_num_list)
        s_list = s_list.to('cpu')

        s_list = s_list*(50/s_list.min())
        s_list = torch.log(s_list)
        s_list = s_list*(1/s_list.min())
        self.s_list = s_list
        self.s = s
        m_list = torch.flip(self.s_list, dims=[0])
        m_list = m_list * (max_m / m_list.max())
        self.m_list = m_list
                
        self.weighted = weighted
        self.weight = weight

    def forward(self, output, target):
        mask = torch.where(target!=-1)
        output = output[mask]
        target = target[mask]

        output = output.cpu()
        target = target.cpu()

        cosine = output*self.s_list
        phi = cosine - self.m_list

        index = torch.zeros_like(output, dtype=torch.uint8)
        index = index.long()
        index.scatter_(1, target.data.view(-1, 1).long(), 1)

        index = index.bool()
        output2 = torch.where(index, phi, cosine)

        if self.weighted == False:
            output2 *= self.s
        elif self.weighted == True:
            index_float = index.type(torch.cuda.FloatTensor)
            batch_s = torch.flip(self.s_list, dims=[0])*self.s
            batch_s = torch.clamp(batch_s, self.s, 50)
            batch_s = batch_s.to('cpu')
            index_float = index_float.to('cpu')

            batch_s = torch.matmul(batch_s[None, :], index_float.transpose(0,1))
            batch_s = batch_s.view((-1, 1))
            output2 *= batch_s
        else:
            output2 *= self.s

        target = target.long()
        return F.cross_entropy(output2, target, weight=self.weight)

class CombinedLoss(nn.Module):
    def __init__(self, soft_dice_params, kps_params, lambda_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.soft_dice_loss = SoftDICELoss(**soft_dice_params)
        self.kps_loss = KPSLoss(**kps_params)
        self.lambda_weight = lambda_weight

    def forward(self, output, target):
        dice_loss = self.soft_dice_loss(output, target)
        kps_loss = self.kps_loss(output, target)
        combined_loss = (1 - self.lambda_weight) * dice_loss + self.lambda_weight * kps_loss
        return combined_loss
    
    def update(self, lambda_weight):
        self.lambda_weight = lambda_weight

