'''
Portions of this code copyright 2020, princeton-vl 
In the framework of:
Teed, Zachary, and Jia Deng. "Raft: Recurrent all-pairs field transforms for optical flow." European Conference on Computer Vision. Springer, Cham, 2020.
URL: https://github.com/princeton-vl/RAFT
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import Variable
# import torchvision

from .submodules_RAFT_extractor256 import BasicEncoder256
from .submodules_RAFT_GRU256 import BasicUpdateBlock256

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
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
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dy = torch.linspace(-r, r, 2*r+1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)

            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()


    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr  / torch.sqrt(torch.tensor(dim).float())

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
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (i_loss).mean()
        
    flow_loss = flow_loss + l2_loss * factor
    
    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }
    
    return flow_loss, metrics

class LanczosUpsampling(nn.Module):
    """
    Lanczos4 upsampling module
    """

    def __init__(self, img_shape, new_size, a=4):
        super(LanczosUpsampling, self).__init__()
        torch.pi = torch.acos(torch.zeros(1)).item() * 2
        delta_X = torch.range(0. + 1e-8, 1., 1. / 1024)
        self.B, self.C, self.H, self.W = img_shape
        self.new_size = new_size

        self.lanczos_kernel = torch.stack(((torch.sin((delta_X - 4) * torch.pi) / ((delta_X - 4) * torch.pi)) * (torch.sin((delta_X - 4) / a * torch.pi) / ((delta_X - 4) / a * torch.pi)), \
                                           (torch.sin((delta_X - 3) * torch.pi) / ((delta_X - 3) * torch.pi)) * (torch.sin((delta_X - 3) / a * torch.pi) / ((delta_X - 3) / a * torch.pi)), \
                                           (torch.sin((delta_X - 2) * torch.pi) / ((delta_X - 2) * torch.pi)) * (torch.sin((delta_X - 2) / a * torch.pi) / ((delta_X - 2) / a * torch.pi)), \
                                           (torch.sin((delta_X - 1) * torch.pi) / ((delta_X - 1) * torch.pi)) * (torch.sin((delta_X - 1) / a * torch.pi) / ((delta_X - 1) / a * torch.pi)), \
                                           (torch.sin((delta_X) * torch.pi) / ((delta_X) * torch.pi)) * (torch.sin((delta_X) / a * torch.pi) / ((delta_X) / a * torch.pi)), \
                                           (torch.sin((delta_X + 1) * torch.pi) / ((delta_X + 1) * torch.pi)) * (torch.sin((delta_X + 1) / a * torch.pi) / ((delta_X + 1) / a * torch.pi)), \
                                           (torch.sin((delta_X + 2) * torch.pi) / ((delta_X + 2) * torch.pi)) * (torch.sin((delta_X + 2) / a * torch.pi) / ((delta_X + 2) / a * torch.pi)), \
                                           (torch.sin((delta_X + 3) * torch.pi) / ((delta_X + 3) * torch.pi)) * (torch.sin((delta_X + 3) / a * torch.pi) / ((delta_X + 3) / a * torch.pi)), \
                                           (torch.sin((delta_X + 4) * torch.pi) / ((delta_X + 4) * torch.pi)) * (torch.sin((delta_X + 4) / a * torch.pi) / ((delta_X + 4) / a * torch.pi))) \
                                          ).cuda()

        self.y_init, self.x_init = torch.meshgrid(torch.arange(0, self.H, 1), torch.arange(0, self.W, 1))
        self.y_new, self.x_new = torch.meshgrid(torch.arange(0, self.H, self.H / self.new_size[2]),
                                                torch.arange(0, self.W, self.W / self.new_size[3]))
        self.y_init_up, self.x_init_up = torch.floor(self.y_new.cuda()).long().cuda(), torch.floor(self.x_new.cuda()).long().cuda()

        self.y_sub, self.x_sub = self.y_new.cuda() - self.y_init_up.cuda(), self.x_new.cuda() - self.x_init_up.cuda()

        self.unfold_x = torch.nn.Unfold(kernel_size=(1, 9))
        self.unfold_y = torch.nn.Unfold(kernel_size=(9, 1))

    def forward(self, img, new_size):
        B, C, H, W = img.shape

        y_init, x_init = torch.meshgrid(torch.arange(0, H, 1), torch.arange(0, W, 1))
        y_new, x_new = torch.meshgrid(torch.arange(0, H, H / self.new_size[2]),
                                      torch.arange(0, W, W / self.new_size[3]))
        y_init_up, x_init_up = torch.floor(y_new.cuda()).long().cuda(), torch.floor(x_new.cuda()).long().cuda()

        y_sub, x_sub = y_new.cuda() - y_init_up.cuda(), x_new.cuda() - x_init_up.cuda()
        img_up_rough = img[:, :, y_init_up, x_init_up]

        ### horizontal shift
        # padding
        p1d = (4, 4, 0, 0)
        padded_img_up = F.pad(img_up_rough, p1d, mode='reflect')
        padded_x_sub = F.pad(torch.unsqueeze(torch.unsqueeze(x_sub, dim=0), dim=0), p1d, mode='reflect')

        # unfold patch
        padded_img_unfold = torch.squeeze(self.unfold_x(padded_img_up))
        padded_x_sub_unfold = torch.squeeze(self.unfold_x(padded_x_sub))

        # compute index and select kernel
        center_point = [4]
        center_index = torch.floor(padded_x_sub_unfold / (1.0 / 1024))[center_point, :]
        x_kernel = self.lanczos_kernel[:, center_index.long()].repeat(C, B, 1).permute(1, 0, 2).cuda()
        x_shifted_patch = torch.sum((x_kernel * padded_img_unfold).reshape(B,C,9,-1),dim=2).reshape(new_size)


        ### vertical shift
        # padding
        p2d = (0, 0, 4, 4)
        padded_img_up = F.pad(x_shifted_patch, p2d, mode='reflect')
        padded_y_sub = F.pad(torch.unsqueeze(torch.unsqueeze(y_sub, dim=0), dim=0), p2d, mode='reflect')

        # unfold patch
        padded_img_unfold = torch.squeeze(self.unfold_y(padded_img_up))
        padded_y_sub_unfold = torch.squeeze(self.unfold_y(padded_y_sub))

        # compute index and select kernel
        center_point = [4]
        center_index = torch.floor(padded_y_sub_unfold / (1.0 / 1024))[center_point, :]
        y_kernel = torch.squeeze(self.lanczos_kernel[:, center_index.long()])
        y_kernel = self.lanczos_kernel[:, center_index.long()].repeat(C, B, 1).permute(1, 0, 2).cuda()
        y_shifted_patch = torch.sum((y_kernel * padded_img_unfold).reshape(B,C,9,-1),dim=2).reshape(new_size)

        return y_shifted_patch

class RAFT256(nn.Module):
    """
    RAFT
    """
    def __init__(self,args):
        super(RAFT256,self).__init__()

        self.hidden_dim = 128
        self.context_dim = 128
        self.corr_levels = 4
        self.corr_radius = 4
        self.flow_size = 32
        
        self.fnet = BasicEncoder256(output_dim=256, norm_fn='instance', dropout=0., threshold=args.channel_threshold)
        self.cnet = BasicEncoder256(output_dim=self.hidden_dim+self.context_dim, norm_fn='instance', dropout=0., threshold=args.channel_threshold)
        self.update_block = BasicUpdateBlock256(hidden_dim=self.hidden_dim, corr_levels=self.corr_levels, corr_radius=self.corr_radius)
        if args.upsample == 'bicubic':
            self.upsample_bicubic = nn.Upsample(scale_factor=2, mode='bicubic')
        elif args.upsample == 'bicubic8':
            self.upsample_bicubic8 = nn.Upsample(scale_factor=8, mode='bicubic')
        elif args.upsample == 'lanczos4':
            self.upsample_lanczos2_1 = LanczosUpsampling([args.batch_size, 2, self.flow_size, self.flow_size], [args.batch_size, 2, self.flow_size * 2, self.flow_size * 2])
        elif args.upsample == 'lanczos4_8':
            self.upsample_lanczos2_2 = LanczosUpsampling([args.batch_size, 2, self.flow_size * 2, self.flow_size * 2], [args.batch_size, 2, self.flow_size * 4, self.flow_size * 4])
            self.upsample_lanczos2_3 = LanczosUpsampling([args.batch_size, 2, self.flow_size * 4, self.flow_size * 4], [args.batch_size, 2, self.flow_size * 8, self.flow_size * 8])
            self.upsample_lanczos8 = LanczosUpsampling([args.batch_size, 2, self.flow_size, self.flow_size], [args.batch_size, 2, self.flow_size * 8, self.flow_size * 8])
    
    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//4, W//4).to(img.device)
        coords1 = coords_grid(N, H//4, W//4).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/4, W/4, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(4 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 4*H, 4*W)

    def forward(self,input,args,flow_init=None):
        img1 = torch.unsqueeze(input[:,0,:,:], dim=1)
        img2 = torch.unsqueeze(input[:,1,:,:], dim=1)

        with autocast(enabled=args.amp):
            fmap1, fmap2 = self.fnet([img1, img2])

        corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius, num_levels=self.corr_levels)

        with autocast(enabled=args.amp):
            cnet = self.cnet(img1)
            net, inp = torch.split(cnet, [self.hidden_dim, self.context_dim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(img1)

        if flow_init is not None:
            flow_init = F.upsample(flow_init, [coords1.size()[2],coords1.size()[3]], mode='bilinear')

            coords1 = coords1 + flow_init
        
        flow_predictions = []

        for itr in range(args.iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)

            flow = coords1 - coords0

            with autocast(enabled=args.amp):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            if args.upsample == 'convex':
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            elif args.upsample == 'bicubic':
                flow_up = self.upsample_bicubic(self.upsample_bicubic(self.upsample_bicubic(coords1 - coords0)))
            elif args.upsample == 'bicubic8':
                flow_up = self.upsample_bicubic8(coords1 - coords0)
            elif args.upsample == 'lanczos4':
                B_f, C_f, H_f, W_f = coords1.shape
                flow_up = self.upsample_lanczos2_1(coords1 - coords0, new_size=[B_f, C_f, H_f * 2, W_f * 2])
                flow_up = self.upsample_lanczos2_2(flow_up, new_size=[B_f, C_f, H_f * 4, W_f * 4])
                flow_up = self.upsample_lanczos2_3(flow_up, new_size=[B_f, C_f, H_f * 8, W_f * 8])
            elif args.upsample == 'lanczos4_8':
                B_f, C_f, H_f, W_f = coords1.shape
                flow_up = self.upsample_lanczos8(coords1 - coords0, new_size=[B_f, C_f, H_f * 8, W_f * 8])
            else:
                raise ValueError('Selected upsample method not supported: ', args.upsample)

            flow_predictions.append(flow_up)
        # loss = sequence_loss(flow_predictions, flowl0)
        if args.test == 1:
            return flow_predictions
        else:
            return flow_predictions,fmap1,fmap2,cnet
