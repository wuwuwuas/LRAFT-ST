'''
Portions of this code copyright 2020, princeton-vl 
In the framework of:
Teed, Zachary, and Jia Deng. "Raft: Recurrent all-pairs field transforms for optical flow." European Conference on Computer Vision. Springer, Cham, 2020.
URL: https://github.com/princeton-vl/RAFT
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from .Extractor_threshold import BasicEncoder
from .Update_GRU import BasicUpdateBlock

try:
    autocast = torch.cuda.amp.autocast
except:

    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()

    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


def sequence_loss(flow_preds, flow_gt, l2, params, factor):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    flow_loss = 0.0
    l2_loss = 0
    if l2:
        for parm in params:
            l2_loss = l2_loss + torch.norm(parm, p=2)

    for i in range(n_predictions):
        i_weight = 0.8**(n_predictions - i - 1)
        flow = flow_gt
        i_loss = (flow_preds[i] - flow).abs()
        flow_loss += i_weight * (i_loss).mean()
    flow_loss = flow_loss + l2_loss * factor

    pre = flow_preds[-1]
    u_gt = flow_gt[:,0,:,:]
    v_gt = flow_gt[:,1,:,:]
    u_pre = pre[:,0,:,:]
    v_pre = pre[:,1,:,:]
    sq = torch.square(u_gt-u_pre) + torch.square(v_gt-v_pre)
    sum_sq = torch.sum(torch.sum(sq, dim = 1), dim = 1)
    mean_sum = sum_sq / 256/256
    rmse = torch.mean(torch.sqrt(mean_sum))
    
    sq = torch.square(u_gt-u_pre) + torch.square(v_gt-v_pre)
    sqq = torch.sqrt(sq)
    aee = torch.mean(sqq)

    metrics = {
        'rmse': rmse.item(),
        'epe': aee.mean().item()
    }

    return flow_loss, metrics

def soft_loss(flow_preds, flow_teacher):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = 0.8**(n_predictions - i - 1)
        flow = flow_teacher[i]
        i_loss = (flow_preds[i] - flow).abs()
        flow_loss += i_weight * (i_loss).mean()

    return flow_loss


def feature_loss(fea1, fea2):
    """ Loss function defined over sequence of flow predictions """

    diff = fea1 - fea2
    loss = torch.norm(diff, p=2)

    return loss

class LRAFT_ST(nn.Module):

    def __init__(self,args):
        super(LRAFT_ST,self).__init__()

        self.hidden_dim1 = 128
        self.context_dim1 = 128
        self.hidden_dim2 = 128
        self.context_dim2 = 128
        self.output_dim1 = self.hidden_dim1 + self.context_dim1
        self.output_dim2 = self.hidden_dim2 + self.context_dim2
        self.corr_levels = 4
        self.corr_radius = 4

        self.fnet = BasicEncoder(output_dim=self.output_dim2, norm_fn='instance', dropout=0., threshold=args.channel_threshold, name = 'fnet1')
        self.cnet = BasicEncoder(output_dim=self.output_dim1, norm_fn='instance', dropout=0., threshold=args.channel_threshold, name = 'cnet1')

        self.update_block = BasicUpdateBlock(hidden_dim=self.hidden_dim2, corr_levels=self.corr_levels,
                                              corr_radius=self.corr_radius)
        self.test = args.test
    def initialize_flow(self, img, scale):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//scale, W//scale).to(img.device)
        coords1 = coords_grid(N, H//scale, W//scale).to(img.device)
        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def desample_image(self,image, num):
        image_pair = []
        image_pair.append(image)
        for i in range(num):
            image = nn.AvgPool2d(kernel_size=2, stride=2)(image)
            # image = F.interpolate(image, scale_factor = 0.5, mode='bilinear') # 'linear', 'bilinear', 'bicubic'
            image_pair.append(image)
        return image_pair

    def forward_wrap(self, img, coords):
        """ Wrapper for grid_sample, uses pixel coordinates """
        H, W = img.shape[-2:]
        # xgrid, ygrid = coords.split([1,1], dim=1)
        xgrid = coords[:, 0:1, :, :]
        ygrid = coords[:, 1:, :, :]
        xgrid = 2 * xgrid / (W - 1) - 1
        ygrid = 2 * ygrid / (H - 1) - 1
        grid = torch.cat([xgrid, ygrid], dim=1)
        img = F.grid_sample(img, grid.permute(0, 2, 3, 1), align_corners=True, mode='bilinear')
        return img
    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(4 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 4*H, 4*W)

    def forward(self,input,args,flow_init=None):
        image1 = input[:,0:1,:,:]
        image2 = input[:,1:,:,:]

        # 1
        fmap1 = self.fnet(image1)
        fmap2 = self.fnet(image2)

        cmap1 = self.cnet(image1)

        corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius, num_levels=self.corr_levels)

        net, inp = torch.split(cmap1, [self.hidden_dim2, self.context_dim2], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1, self.corr_levels)
        flow_predictions=[]

	
        for itr in range(args.iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume
            flow = coords1 - coords0
            with autocast(enabled=args.amp):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
            coords1 = coords1 + delta_flow
            flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)
        if self.test == 1:
            return flow_predictions
        else:
            return flow_predictions, fmap1, fmap2, cmap1
