import torch as t
from torch import nn


class Loss(nn.Module):

    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, landmark_pred, euler_angle_pred, landmark_gt, euler_angle_gt, attribute_gt, train_batch_size):
        if landmark_pred.size()[0] < train_batch_size:
            train_batch_size = landmark_pred.size()[0]
        angle_weight = t.sum(1 - t.cos(euler_angle_gt - euler_angle_pred), dim=1)
        mat_ratio = t.mean(attribute_gt, dim=0)
        mat_ratio_more_than_zero = mat_ratio > 0
        mat_ratio_equal_zero = mat_ratio == 0
        mat_ratio[mat_ratio_more_than_zero] = 1 / mat_ratio[mat_ratio > 0]
        mat_ratio[mat_ratio_equal_zero] = train_batch_size
        attribute_weight = t.sum(attribute_gt * mat_ratio, dim=1)
        # attribute_weight = t.sum(attribute_gt.cpu() * t.tensor([1.0 / x if x > 0 else train_batch_size for x in t.mean(attribute_gt, dim=0).cpu()]), dim=1).cuda(0)
        landmark_l2_distance = t.sum((landmark_gt - landmark_pred.view((landmark_pred.size()[0], 98, -1))) ** 2, dim=2)
        weight_loss = t.mean(t.sum(attribute_weight.view((attribute_weight.size()[0], -1)) * angle_weight.view((angle_weight.size()[0], -1)) * landmark_l2_distance, dim=1))
        return weight_loss, t.mean(t.sum(landmark_l2_distance, dim=1))
