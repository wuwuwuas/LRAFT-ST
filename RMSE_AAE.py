# gt:B H w 2
# rmse: root mean square error, rmse
# aae: average angular error
# aee: average ebdpoint error
import torch

def rmse(gt, pre):
    u_gt = gt[:,:,:,0]
    v_gt = gt[:,:,:,1]
    u_pre = pre[:,:,:,0]
    v_pre = pre[:,:,:,1]
    
    sq = torch.square(u_gt-u_pre) + torch.square(v_gt-v_pre)
    sum_sq = torch.sum(torch.sum(sq, dim = 1), dim = 1)
    mean_sum = sum_sq / (gt.size()[1] * gt.size()[2])
    rmse = torch.mean(torch.sqrt(mean_sum))
    
    return rmse


def aee(gt, pre):
    u_gt = gt[:,:,:,0]
    v_gt = gt[:,:,:,1]
    u_pre = pre[:,:,:,0]
    v_pre = pre[:,:,:,1]
    
    sq = torch.square(u_gt-u_pre) + torch.square(v_gt-v_pre)
    sqq = torch.sqrt(sq)
    aee = torch.mean(sqq)
    
    #aee = torch.sum((pre[:, :, :, :] - gt[:, :, :, :]) ** 2,
    #                          dim=3).sqrt().view(-1).mean()
    return aee

