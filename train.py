import os
import argparse
import torch
from tqdm import tqdm
from RMSE_AAE import rmse
import math
import numpy as np
import importlib

import os.path
from torch.utils.data import TensorDataset, DataLoader
import hdf5storage
import time

###############################################################################

###############################################################################

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

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

### main method

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', default=0, type=int,
                        help='index of gpu used')
    parser.add_argument('--name', type=str, default='LightPIVNet',
                        help='name of experiment')
    parser.add_argument('--input_path_ckpt', type=str, default=None,
                        help='path of already trained checkpoint')
    parser.add_argument('--recover', type=eval, default=False,
                        help='Wether to load an existing checkpoint')

    parser.add_argument('--kd', type=eval, default=False,
                        help='Wether to use knowledge distillation')
    parser.add_argument('--input_teacher', type=str, default=None,
                        help='path of the pre-trained teacher model')

    parser.add_argument('--output_dir_ckpt', type=str, default='./checkpoints/',
                        help='output directory of checkpoint')
    parser.add_argument('--channel_threshold', type=str, default=False,
                        choices=['False', 'True'],
                        help='Wether to use channel wise threshold')
    parser.add_argument('--dataset', type=str, default='dataset1',
                        choices=['dataset1', 'dataset2'],
                        help='dataset for train')
    parser.add_argument('--test', type=int, default=0, choices=[0, 1],
                        help='whether the evaluation mode is used, 1 yes, 0 no')
    parser.add_argument('--amp', type=eval, default=False,
                        help='Wether to use auto mixed precision')
    parser.add_argument('-a', '--arch', type=str, default='LRAFT-ST',
                        choices=['RAFT_4', 'RAFT_4-ST', 'LRAFT', 'LRAFT-ST'],
                        help='Type of architecture to use')
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--init_lr', default=0.0001, type=float,
                        help='initial learning rate')
    parser.add_argument('--reduce_factor', default=0.5, type=float,
                        help='reduce factor of ReduceLROnPlateau scheme')
    parser.add_argument('--patience_level', default=5, type=int,
                        help='patience level of ReduceLROnPlateau scheme')
    parser.add_argument('--min_lr', default=1e-8, type=float,
                        help='minimum learning rate')
    parser.add_argument('--iters', default=12, type=int,
                        help='number of update steps in ConvGRU')
    parser.add_argument('--upsample', type=str, default='convex',
                        choices=['convex'],
                        help="""Type of upsampling method""")

    parser.add_argument('--l2', type=str, default=True,
                        help='Wether to apply l2 regularisation to the network parameters')
    parser.add_argument('--l2_factor', type=float, default=0.0001,
                        help='factor of the l2 loss')
    parser.add_argument('--ls', type=str, default=False,
                        help='Wether to use the soft loss')
    parser.add_argument('--ls_factor', type=float, default=0.1,
                        help='factor of the soft loss')
    parser.add_argument('--lf', type=str, default=False,
                        help='Wether to use the feature loss')
    parser.add_argument('--lf_factor', type=float, default=0.0001,
                        help='factor of the feature loss')
    args = parser.parse_args()
    print('args parsed')

    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    train(args)
        
    
def train(args):
    device = torch.device(f'cuda:{args.gpu}')
    if args.arch == 'LRAFT-ST' or args.arch == 'LRAFT':
        module_name = 'LRAFT-ST.RAFT_threshold'
        model_name = 'LRAFT_ST'
        loss_name = 'sequence_loss'
        module = importlib.import_module(module_name)
        LRAFT_ST = getattr(module, model_name)
        sequence_loss = getattr(module, loss_name)
        loss_name = 'soft_loss'
        soft_loss = getattr(module, loss_name)
        loss_name = 'feature_loss'
        feature_loss = getattr(module, loss_name)
        model = LRAFT_ST(args)

        if args.kd:
            module_name = 'RAFT_4-ST.flowNetsRAFT256'
            model_name = 'RAFT256'
            module = importlib.import_module(module_name)
            RAFT256 = getattr(module, model_name)
            model_teacher = RAFT256(args)
            checkpoint = torch.load(args.input_teacher)
            model_teacher.load_state_dict(checkpoint['model_state_dict'])
            model_teacher.to(device)
        print('Selected model: LRAFT - -', args.arch, '  Channel threshold- -',args.channel_threshold,
              '  knowledge distillation- -', args.kd, '  Soft loss', args.ls,'  Feature loss', args.lf)
        
    elif args.arch == 'RAFT_4' or args.arch == 'RAFT_4-ST':
        module_name = 'RAFT_4-ST.flowNetsRAFT256'
        model_name = 'RAFT256'
        loss_name = 'sequence_loss'
        module = importlib.import_module(module_name)
        RAFT256 = getattr(module, model_name)
        sequence_loss = getattr(module, loss_name)
        model = RAFT256(args)
        print('Selected model: RAFT 1/4 resolution- -', args.arch, '  Channel threshold- -',args.channel_threshold)

    else:
        raise ValueError('Selected model not supported: ', args.arch)

    checkpoint_dir = args.output_dir_ckpt + args.name
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = checkpoint_dir + '/ckpt.tar'

    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of trainable parameters: ', pytorch_trainable_params)


    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.reduce_factor, patience=args.patience_level, min_lr=args.min_lr)
    start_epoch = 0
    lowest_val_epe = 100000.0
    
    if args.recover:
        print('recovering: ', args.input_path_ckpt)
        checkpoint = torch.load(args.input_path_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print('model recovered')
        checkpoint = torch.load(args.input_path_ckpt)
        start_epoch = checkpoint['epoch']+1
        lowest_val_rmse = checkpoint['rmse']
        print("recovering at epoch: ", start_epoch)


    if args.dataset == 'dataset1':
        train_image_path = './data/Dataset1/train_image_100.mat'
        train_flow_path = './data/Dataset1/train_flow_100.mat'
        test_image_path = './data/Dataset1/test_image_20.mat'
        test_flow_path = './data/Dataset1/test_flow_20.mat'
        
    elif args.dataset == 'dataset2':
        train_image_path = './data/Dataset2/train_image_100.mat'
        train_flow_path = './data/Dataset2/train_flow_100.mat'
        test_image_path = './data/Dataset2/test_image_20.mat'
        test_flow_path = './data/Dataset2/test_flow_20.mat'
    else:
        raise ValueError('Selected dataset not supported: ', args.dataset)

    train_image = hdf5storage.loadmat(train_image_path)
    train_flow = hdf5storage.loadmat(train_flow_path)
    test_image = hdf5storage.loadmat(test_image_path)
    test_flow = hdf5storage.loadmat(test_flow_path)

    train_image = torch.from_numpy(train_image['image']).to(torch.float32) / 255
    train_flow = torch.from_numpy(train_flow['flow']).to(torch.float32)
    test_image = torch.from_numpy(test_image['image']).to(torch.float32) / 255
    test_flow = torch.from_numpy(test_flow['flow']).to(torch.float32)
    
    train_data = TensorDataset(train_image[:,:,:,:],train_flow[:,:,:,:])
    test_data = TensorDataset(test_image[:,:,:,:],test_flow[:,:,:,:])
    
    print(train_flow.shape)
    train_data = DataLoader(train_data, batch_size = args.batch_size, shuffle = True)
    test_data = DataLoader(test_data, batch_size = args.batch_size, shuffle = True)

    ep_stop = 0
    for epoch in range(start_epoch, args.epochs, 1):
        model.train()
        last_training_loss, last_validation_loss = 0.0, 0.0
        sum_training_loss, sum_validation_loss = 0.0, 0.0
        total_training_samples, total_validation_samples = 0, 0
        sum_training_epe, sum_validation_epe = 0.0, 0.0
        epoch_train_rmse = []
        epoch_test_rmse = []
        train_loader_len = int(math.ceil(len(train_data)))
        train_pbar = tqdm(enumerate(train_data), total=train_loader_len,
                          desc='Epoch: [' + str(epoch + 1) + '/' + str(args.epochs) + '] Training',
                          postfix='loss: ' + str(last_training_loss), position=0, leave=False)
        start_time = time.time()
        #Training
        for i, sample_batched in train_pbar:
            images, flows = sample_batched

            images = images.to(device)
            flows = flows.to(device)
            if args.kd:
                with torch.set_grad_enabled(False):
                    pred_flows_teacher, fmap1_teacher, fmap2_teacher, cmap1_teacher = model_teacher(images, args=args)

            with torch.set_grad_enabled(True):    
                with autocast(enabled=args.amp):
                    if args.arch == 'LRAFT-ST' or args.arch == 'LRAFT' or args.arch == 'RAFT_4-ST' or args.arch == 'RAFT_4':
                        pred_flows, fmap1_stu, fmap2_stu, cmap1_stu = model(images, args=args)
                    else:
                        pred_flows = model(images, args=args)
                    if args.l2:
                        params = list(model.parameters())
                        training_loss, metrics = sequence_loss(pred_flows, flows, l2=args.l2, params=params,
                                                               factor=args.l2_factor)
                    else:
                        training_loss, metrics = sequence_loss(pred_flows, flows, l2=False, params=0, factor=0)
    
                    train_epe_loss = metrics['epe']
                    fflows = flows.permute(0, 2, 3, 1)
                    pred_fflows = pred_flows[-1].permute(0, 2, 3, 1)
                    train_rmse = rmse(fflows, pred_fflows)
    
                    epoch_train_rmse.append(train_rmse.item())

                    training_loss_soft = 0
                    training_loss_fea = 0
                    if args.ls:
                        training_loss_soft = soft_loss(pred_flows , pred_flows_teacher)
                    if args.lf:
                        training_loss_fea = feature_loss(fmap1_teacher, fmap1_stu) + feature_loss(fmap2_teacher, fmap2_stu) + feature_loss(cmap1_teacher, cmap1_stu)
                    training_loss = training_loss + training_loss_soft * args.ls_factor + args.lf_factor * training_loss_fea
                    
                    
                    sum_training_loss += training_loss.item() * images.shape[0]
                    total_training_samples += images.shape[0]
                    epoch_training_loss = sum_training_loss / total_training_samples
    
                    sum_training_epe += train_epe_loss * images.shape[0]
                    epoch_train_epe_loss = sum_training_epe / total_training_samples
    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
                    training_loss.backward()
                    optimizer.step()
    
                    train_pbar.set_postfix_str(
                        'loss: ' + "{:10.6f}".format(epoch_training_loss) + \
                        ' epe: ' + "{:10.6f}".format(epoch_train_epe_loss))
        epoch_train_rmse = np.mean(epoch_train_rmse)
        # Validation
        with torch.set_grad_enabled(False):
            #set evaluation mode
            model.eval()

            val_loader_len = int(math.ceil(len(test_data)))
            val_pbar = tqdm(enumerate(test_data), total=val_loader_len,
                        desc='Epoch: [' + str(epoch+1) + '/' + str(args.epochs) + '] Validation',
                        postfix='loss: ' + str(last_validation_loss), position=1, leave=False)

            for i, sample_batched in val_pbar:
                images, flows = sample_batched
                
                images = images.to(device)
                flows = flows.to(device)
                
                with autocast(enabled=args.amp):
                    if args.arch == 'LRAFT-ST' or args.arch == 'LRAFT' or args.arch == 'RAFT_4-ST' or args.arch == 'RAFT_4':
                        pred_flows, fmap1_stu, fmap2_stu, cmap1_stu = model(images, args=args)
                    else:
                        pred_flows = model(images, args=args)
                    
                    if args.l2:
                        params = list(model.parameters())
                        validation_loss, metrics = sequence_loss(pred_flows, flows, l2=args.l2, params=params, factor=args.l2_factor)
                    else:
                        validation_loss, metrics = sequence_loss(pred_flows, flows, l2=False, params=0, factor=0)

                    val_epe_loss = metrics['epe']

                    fflows = flows.permute(0, 2, 3, 1)
                    pred_fflows = pred_flows[-1].permute(0, 2, 3, 1)
                    test_rmse = rmse(fflows, pred_fflows)
                    
                    epoch_test_rmse.append(test_rmse.item())
                    sum_validation_loss += validation_loss.item() * images.shape[0]
                    total_validation_samples += images.shape[0]
                    epoch_validation_loss = sum_validation_loss / total_validation_samples

                    sum_validation_epe += val_epe_loss * images.shape[0]
                    epoch_val_epe_loss = sum_validation_epe / total_validation_samples

                    val_pbar.set_postfix_str(
                        'loss: ' + "{:10.6f}".format(epoch_validation_loss) + \
                        ' epe: ' + "{:10.6f}".format(epoch_val_epe_loss))
        epoch_test_rmse = np.mean(epoch_test_rmse)
        scheduler.step(epoch_val_epe_loss)
        metrics_tensor = torch.tensor(
            [epoch_training_loss, epoch_validation_loss, epoch_train_epe_loss, epoch_val_epe_loss, epoch_train_rmse, epoch_test_rmse])
        end_time = time.time()
        times = end_time - start_time

        loss_metric = metrics_tensor
        print('Epoch: ', epoch + 1, 'time: ', times, ' training loss: ', loss_metric[0].item(), ' validation loss: ', \
              loss_metric[1].item(), ' train epe: ', loss_metric[2].item(), ' val epe: ', loss_metric[3].item(), \
              'train_rmse: ',loss_metric[4].item(), 'test_rmse: ', loss_metric[5].item(),  'lr: ', str(optimizer.__getattribute__('param_groups')[0]['lr']), flush=True)

        ep_stop = ep_stop + 1
        if ep_stop > 100:
            print('early stop')
            break
        # save model
        if (loss_metric[5] < lowest_val_epe):
            ep_stop = 0
            lowest_val_epe = loss_metric[5]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'epe': lowest_val_epe
            }, checkpoint_path)
            print('model saved and lowest epe overwritten:', lowest_val_epe, flush=True)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
