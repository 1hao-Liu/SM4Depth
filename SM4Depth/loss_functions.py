import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from pytorch3d.loss import chamfer_distance

import numpy as np


# silog + gradient loss
class classbalanced_refine_loss(nn.Module):
    def __init__(self, variance_focus=0.85, gradient_scales=4, cls_num=4):
        super().__init__()
        self.variance_focus = variance_focus
        self.gradient_scales = gradient_scales
        self.cls_num = cls_num

    def forward(self, prediction, target, depth_type):
        losses_gradient=[]
        losses_silog=[]
        cls_losses=torch.zeros(self.cls_num).cuda()
        cls_cnt=torch.zeros(self.cls_num).cuda()

        # compute loss for each resolution
        for idx in range(len(prediction)):
            loss_gradient = 0
            pd = prediction[idx].squeeze(1)
            gt = target[idx].squeeze(1)
            mask = gt > 0.1

            for scale in range(self.gradient_scales):
                step = pow(2, scale) 
                loss_gradient += GradientLoss(pd[:, ::step, ::step], gt[:, ::step, ::step], mask[:, ::step, ::step])
            
            # compute silog for the 1st, 2nd, 3th depth predictions
            if idx < len(prediction)-1:
                d = torch.log(pd[mask]) - torch.log(gt[mask])
                loss_silog = torch.sqrt((d ** 2).mean() - 0.85 * (d.mean() ** 2)) * 10.0
            # compute silog for the last depth prediction
            else:
                loss_silog = torch.tensor(0).float().cuda()
                for f1_idx in range(pd.shape[0]):
                    d = torch.log(pd[f1_idx][mask[f1_idx]]) - torch.log(gt[f1_idx][mask[f1_idx]])
                    d = torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0
                    if not torch.isinf(d) and not torch.isnan(d):
                        loss_silog += d
                        cls_losses[depth_type[f1_idx]] += d
                    cls_cnt[depth_type[f1_idx]] += 1

                loss_silog /= pd.shape[0]
                avgcls_meansilog = cls_losses / cls_cnt 

            losses_gradient.append(loss_gradient)
            losses_silog.append(1e-3 if torch.isnan(loss_silog) else loss_silog)
        
        total_gradient = sum(losses_gradient)
        total_silog = sum(losses_silog)

        return total_gradient, total_silog, avgcls_meansilog
    

class BalanceClassRatio(nn.Module):
    def __init__(self, batchsize, cls_num):
        super().__init__()
        self.batchsize = batchsize
        self.cls_num = cls_num
    
    def forward(self, loss):
        cls_weight = [int(self.batchsize/self.cls_num)] * self.cls_num  # [2, 2, 2, 2]
        allocated = self.batchsize - sum(cls_weight)  # 2
        avgcls_meanloss = loss.clone().detach()
        avgcls_meanloss[torch.argmin(avgcls_meanloss)] = 0  # del min loss
        avgcls_meanloss /= sum(avgcls_meanloss)  # get losses' ratio
        for i in range(allocated):
            cls_weight[torch.argmax(avgcls_meanloss)] += 1
            avgcls_meanloss[torch.argmax(avgcls_meanloss)] /= 2
            avgcls_meanloss /= sum(avgcls_meanloss)
        
        return cls_weight


def GradientLoss(prediction, target, mask):
        M = torch.sum(mask, (1, 2))
        diff = prediction - target
        diff = torch.mul(mask, diff)
        grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
        mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
        grad_x = torch.mul(mask_x, grad_x)
        grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
        mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
        grad_y = torch.mul(mask_y, grad_y)

        image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

        divisor = torch.sum(M) * 2
        if divisor == 0:
            return 0
        else:
            return torch.sum(image_loss) / divisor


# chamfer loss
class chamfer_loss(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.param = param

    def forward(self, bins, target_depth_maps):
        input_points = bins.squeeze(-1)
        target_points = target_depth_maps.flatten(1)
        mask = target_points.ge(0.1)
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2) 
        loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)

        return self.param * loss


# cross entropy loss
class crossentropy_loss(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.param = param
        self.loss = nn.CrossEntropyLoss()

    def forward(self, depth_type, cls_weight):
        return self.param * self.loss(cls_weight, depth_type)


class VNL_Loss(nn.Module):
    """
    Virtual Normal Loss Function.
    """
    def __init__(self, focal_x, focal_y, input_size,
                 delta_cos=0.867, delta_diff_x=0.01,
                 delta_diff_y=0.01, delta_diff_z=0.01,
                 delta_z=0.0001, sample_ratio=0.15, 
                 param_vnl=5):
        super().__init__()
        self.fx = torch.tensor([focal_x], dtype=torch.float32).cuda()
        self.fy = torch.tensor([focal_y], dtype=torch.float32).cuda()
        self.input_size = input_size
        self.u0 = torch.tensor(input_size[1] // 2, dtype=torch.float32).cuda()
        self.v0 = torch.tensor(input_size[0] // 2, dtype=torch.float32).cuda()
        self.init_image_coor()
        self.delta_cos = delta_cos
        self.delta_diff_x = delta_diff_x
        self.delta_diff_y = delta_diff_y
        self.delta_diff_z = delta_diff_z
        self.delta_z = delta_z
        self.sample_ratio = sample_ratio
        self.param_vnl = param_vnl

    def init_image_coor(self):
        x_row = np.arange(0, self.input_size[1])
        x = np.tile(x_row, (self.input_size[0], 1))
        x = x[np.newaxis, :, :]
        x = x.astype(np.float32)
        x = torch.from_numpy(x.copy()).cuda()
        self.u_u0 = x - self.u0

        y_col = np.arange(0, self.input_size[0])  # y_col = np.arange(0, height)
        y = np.tile(y_col, (self.input_size[1], 1)).T
        y = y[np.newaxis, :, :]
        y = y.astype(np.float32)
        y = torch.from_numpy(y.copy()).cuda()
        self.v_v0 = y - self.v0

    def transfer_xyz(self, depth):
        x = self.u_u0 * torch.abs(depth) / self.fx
        y = self.v_v0 * torch.abs(depth) / self.fy
        z = depth
        pw = torch.cat([x, y, z], 1).permute(0, 2, 3, 1) # [b, h, w, c]
        return pw

    def select_index(self):
        valid_width = self.input_size[1]
        valid_height = self.input_size[0]
        num = valid_width * valid_height   
        p1 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p1)
        p2 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p2)
        p3 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p3)

        p1_x = p1 % self.input_size[1]
        p1_y = (p1 / self.input_size[1]).astype(np.int16)

        p2_x = p2 % self.input_size[1]
        p2_y = (p2 / self.input_size[1]).astype(np.int16)

        p3_x = p3 % self.input_size[1]
        p3_y = (p3 / self.input_size[1]).astype(np.int16)
        p123 = {'p1_x': p1_x, 'p1_y': p1_y, 'p2_x': p2_x, 'p2_y': p2_y, 'p3_x': p3_x, 'p3_y': p3_y}
        return p123

    def form_pw_groups(self, p123, pw):
        """
        Form 3D points groups, with 3 points in each group.
        :param p123: points index
        :param pw: 3D points  [b h w 3]
        :return:
        """
        p1_x = p123['p1_x']
        p1_y = p123['p1_y']
        p2_x = p123['p2_x']
        p2_y = p123['p2_y']
        p3_x = p123['p3_x']
        p3_y = p123['p3_y']

        pw1 = pw[:, p1_y, p1_x, :]
        pw2 = pw[:, p2_y, p2_x, :]
        pw3 = pw[:, p3_y, p3_x, :]
        # [B, N, 3(x,y,z), 3(p1,p2,p3)]
        pw_groups = torch.cat([pw1[:, :, :, np.newaxis], pw2[:, :, :, np.newaxis], pw3[:, :, :, np.newaxis]], 3)
        return pw_groups

    def filter_mask(self, p123, gt_xyz, delta_cos=0.867,
                    delta_diff_x=0.005, delta_diff_y=0.005, delta_diff_z=0.005):
        pw = self.form_pw_groups(p123, gt_xyz)
        pw12 = pw[:, :, :, 1] - pw[:, :, :, 0]
        pw13 = pw[:, :, :, 2] - pw[:, :, :, 0]
        pw23 = pw[:, :, :, 2] - pw[:, :, :, 1]
        ### ignore linear
        pw_diff = torch.cat([pw12[:, :, :, np.newaxis], pw13[:, :, :, np.newaxis], pw23[:, :, :, np.newaxis]], 
                            3)  # [b, n, 3, 3]
        m_batchsize, groups, coords, index = pw_diff.shape
        # [b*n, 3(pw12,pw13,pw23), 3(x,y,z)]
        proj_query = pw_diff.view(m_batchsize * groups, -1, index).permute(0, 2, 1)
        # [b*n, 3(x,y,z), 3(pw12,pw13,pw23)]
        proj_key = pw_diff.view(m_batchsize * groups, -1, index)
        q_norm = proj_query.norm(2, dim=2)
        nm = torch.bmm(q_norm.view(m_batchsize * groups, index, 1), q_norm.view(m_batchsize * groups, 1, index))
        energy = torch.bmm(proj_query, proj_key)  # transpose check [bn, 3(p123), 3(p123)]
        norm_energy = energy / (nm + 1e-8)
        norm_energy = norm_energy.view(m_batchsize * groups, -1)
        mask_cos = torch.sum((norm_energy > delta_cos) + (norm_energy < -delta_cos), 1) > 3
        mask_cos = mask_cos.view(m_batchsize, groups)
        ### ignore padding and invilid depth
        mask_pad = torch.sum(pw[:, :, 2, :] > self.delta_z, 2) == 3

        ## ignore near
        mask_x = torch.sum(torch.abs(pw_diff[:, :, 0, :]) < delta_diff_x, 2) > 0
        mask_y = torch.sum(torch.abs(pw_diff[:, :, 1, :]) < delta_diff_y, 2) > 0
        mask_z = torch.sum(torch.abs(pw_diff[:, :, 2, :]) < delta_diff_z, 2) > 0

        mask_ignore = (mask_x & mask_y & mask_z) | mask_cos
        mask_near = ~mask_ignore
        mask = mask_pad & mask_near

        return mask, pw

    def select_points_groups(self, gt_depth, pred_depth):
        pw_gt = self.transfer_xyz(gt_depth)
        pw_pred = self.transfer_xyz(pred_depth)
        B, C, H, W = gt_depth.shape
        p123 = self.select_index()
        # mask:[b, n], pw_groups_gt: [b, n, 3(x,y,z), 3(p1,p2,p3)]
        mask, pw_groups_gt = self.filter_mask(p123, pw_gt,
                                              delta_cos=0.867,
                                              delta_diff_x=0.005,
                                              delta_diff_y=0.005,
                                              delta_diff_z=0.005)

        # [b, n, 3, 3]
        pw_groups_pred = self.form_pw_groups(p123, pw_pred)
        pw_groups_pred[pw_groups_pred[:, :, 2, :] == 0] = 0.0001
        mask_broadcast = mask.repeat(1, 9).reshape(B, 3, 3, -1).permute(0, 3, 1, 2)
        pw_groups_pred_not_ignore = pw_groups_pred[mask_broadcast].reshape(1, -1, 3, 3)
        pw_groups_gt_not_ignore = pw_groups_gt[mask_broadcast].reshape(1, -1, 3, 3)

        return pw_groups_gt_not_ignore, pw_groups_pred_not_ignore

    def forward(self, gt_depth, pred_depth, select=True):
        """
        Virtual normal loss.
        :param pred_depth: predicted depth map, [B, W, H, C]
        :param data: target label, ground truth depth, [B, W, H, C], padding region [padding_up, padding_down]
        :return:
        """
        gt_points, dt_points = self.select_points_groups(gt_depth, pred_depth)

        gt_p12 = gt_points[:, :, :, 1] - gt_points[:, :, :, 0]
        gt_p13 = gt_points[:, :, :, 2] - gt_points[:, :, :, 0]
        dt_p12 = dt_points[:, :, :, 1] - dt_points[:, :, :, 0]
        dt_p13 = dt_points[:, :, :, 2] - dt_points[:, :, :, 0]

        gt_normal = torch.cross(gt_p12, gt_p13, dim=2)
        dt_normal = torch.cross(dt_p12, dt_p13, dim=2)
        dt_norm = torch.norm(dt_normal, 2, dim=2, keepdim=True)
        gt_norm = torch.norm(gt_normal, 2, dim=2, keepdim=True)
        dt_mask = dt_norm == 0.0
        gt_mask = gt_norm == 0.0
        dt_mask = dt_mask.to(torch.float32)
        gt_mask = gt_mask.to(torch.float32)
        dt_mask *= 0.01
        gt_mask *= 0.01
        gt_norm = gt_norm + gt_mask
        dt_norm = dt_norm + dt_mask
        gt_normal = gt_normal / gt_norm
        dt_normal = dt_normal / dt_norm

        loss = torch.abs(gt_normal - dt_normal)
        loss = torch.sum(torch.sum(loss, dim=2), dim=0)
        if select:
            loss, indices = torch.sort(loss, dim=0, descending=False)
            loss = loss[int(loss.size(0) * 0.25):]
        loss = torch.mean(loss)

        return self.param_vnl*loss
