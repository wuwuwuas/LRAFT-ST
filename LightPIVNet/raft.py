import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock, SmallUpdateBlock
from .extractor import BasicEncoder, SmallEncoder
from .corr import CorrBlock
from .utils import bilinear_sampler, coords_grid, upflow4

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

    # sq = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1)
    # sq = sq.view(-1)
    # sq = torch.sum(sq,dim=1)
    # rmse = sq.mean().item(),
    rmse = torch.mean(torch.sqrt(mean_sum))

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)

    metrics = {
        'rmse': rmse.item(),
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

class LightPIVNet(nn.Module):
    def __init__(self, args):
        super(LightPIVNet, self).__init__()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        if 'dropout' not in args._get_kwargs():
            args.dropout = 0

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='instance', dropout=args.dropout, name='feature')
       # self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout, name='context')
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

####

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

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
        mask = mask.view(N, 1, 9, 4, 4, H, W) # #####
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(4 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 4*H, 4*W) # #####


    def forward(self, input, args, flow_init=None):
        """ Estimate optical flow between pair of frames """

        image1 = input[:,0:1,:,:]
        image2 = input[:,1:,:,:]
        
        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        
        fmap1, fmap2 = self.fnet([image1, image2])
        net, inp = torch.split(fmap1, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(args.iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow4(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        return flow_predictions
