import importlib
import os
import time
import math
import copy

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from collections import defaultdict
import numpy as np
import random
import wandb

import ComputePostBN
from utils.meters import AverageMeter, accuracy
from utils.config import FLAGS
from utils.datasets import get_dataset
from utils.loss_ops import *
from models.new_group_level_ops import *

def get_model():
    model_lib = importlib.import_module(FLAGS.model)
    model = model_lib.Model(FLAGS.num_classes, FLAGS.image_size)

    return model

def set_random_seed(seed=None):
    if seed is None:
        seed = getattr(FLAGS, 'random_seed', 0)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def _train_one_epoch(model, loader, criterion, soft_criterion, optimizer, epoch, lr_scheduler):
    model.train()
    accum_grad = {}
    grad_history = defaultdict(list)
    temp_history = {}
    t_start = time.time()
    for batch_idx, (input, target) in enumerate(loader):
        get_dense = True
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        density_list = [max(FLAGS.density_list), min(FLAGS.density_list)] + random.sample(FLAGS.density_list, 2)
        density_list.sort(reverse=True)

        for density in density_list:
            level = FLAGS.density_list.index(density)
            Pruner(model, FLAGS.pruner, density)
            output = model(input)

            if density == 1.0:
                loss = torch.mean(criterion(output, target))
                soft_target = torch.nn.functional.softmax(output, dim=1)

            else:
                loss = torch.mean(soft_criterion(output, soft_target.detach()))
            loss.backward()
            

            if batch_idx % FLAGS.print_freq == 0 or batch_idx == len(loader) - 1:
                if get_dense == True:
                    get_dense = False
                    if dist.get_rank() == 0:
                        for name, p in model.named_parameters():
                            if p.grad is None:
                                continue
                            g_data = p.grad.view(-1).clone().detach()
                            accum_grad[name] = g_data
                            temp_history[name] = [g_data,]
                else:
                    if dist.get_rank() == 0:
                        for name, p in model.named_parameters():
                            if p.grad is None:
                                continue
                            g_data = p.grad.view(-1).clone().detach() - accum_grad[name]
                            accum_grad[name].copy_(p.grad.view(-1).detach())
                            temp_history[name].append(g_data)


        optimizer.step()
        lr_scheduler.step()

        if batch_idx % FLAGS.print_freq == 0 or batch_idx == len(loader) - 1:
            if dist.get_rank() == 0:
                for name, tensor_list in temp_history.items():
                    grad_history[name].append(torch.stack(tensor_list))
                
            with torch.no_grad():
                for density in sorted(FLAGS.density_list, reverse=True):
                    Pruner(model, FLAGS.pruner, density)
                    batch_size = target.size()[0]
                    output = model(input)
                    loss = torch.mean(criterion(output, target))
                    acc1, _ = accuracy(output, target, topk=(1, 5))
                    corr1, loss = acc1 * batch_size, loss * batch_size
                    stats = torch.tensor([corr1, loss, batch_size], device=dist.get_rank())
                    dist.barrier()
                    dist.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
                    corr1, loss, batch_size = stats.tolist()
                    acc1, loss = corr1 / batch_size, loss / batch_size
                    if dist.get_rank() == 0:
                        if density in  [0.975, 0.7, 0.3, 0.2, 0.175, 0.15, 0.125, 0.1]:
                            for layer in model.module.modules():
                                if isinstance(layer, DynamicGroupConv2d):
                                    mask = layer.mask.cpu().detach()
                                    sp = np.count_nonzero(mask) / np.prod(mask.size())
                                    print(f"{layer.idx} = {100 * sp:.5f}")
                        print(f"TRAIN {time.time() - t_start:.1f}s Epoch: {epoch}/{FLAGS.num_epochs} density=x{density} Acc={acc1} => Loss {loss} ")
    for name, tensor_list in grad_history.items():
        grad_history[name] = torch.stack(grad_history[name]).cpu().detach().numpy()
    return grad_history    

def train_one_epoch(model, loader, criterion, soft_criterion, optimizer, epoch, lr_scheduler):
    model.train()
    t_start = time.time()
    density_list = [min(FLAGS.density_list)] + random.sample(FLAGS.density_list, 2)
    density_list.sort(reverse=True)

    for batch_idx, (input_list, target) in enumerate(loader):
        target = target.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        
        Pruner(model, FLAGS.pruner, 1.0)
        max_output = model(input_list[0].cuda())
        loss = criterion(max_output, target)
        loss.backward()
        
        with torch.no_grad():
            max_output_detach = max_output.clone().detach()

        for density in density_list:
            Pruner(model, FLAGS.pruner, density)
            output = model(input_list[random.randint(0, 3)].cuda())
            loss = soft_criterion(output, max_output_detach)
            loss.backward()

        optimizer.step()
        lr_scheduler.step()

        if batch_idx % FLAGS.print_freq == 0 or batch_idx == len(loader) - 1:
            with torch.no_grad():
                for density in sorted(FLAGS.density_list, reverse=True):
                    Pruner(model, FLAGS.pruner, density)
                    batch_size = target.size()[0]
                    output = model(input_list[0].cuda())
                    loss = criterion(output,target)
                    acc1, _ = accuracy(output, target, topk=(1, 5))
                    corr1, loss = acc1 * batch_size, loss * batch_size
                    stats = torch.tensor([corr1, loss, batch_size], device=dist.get_rank())
                    dist.barrier()
                    dist.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
                    corr1, loss, batch_size = stats.tolist()
                    acc1, loss = corr1 / batch_size, loss / batch_size
                    if dist.get_rank() == 0:
                        print(f"TRAIN {time.time() - t_start:.1f}s Epoch: {epoch}/{FLAGS.num_epochs} density=x{density} Acc={acc1} => Loss {loss} ")


def validate(model, loader, criterion, epoch, post_loader, density_list):
    if dist.get_rank() == 0:
        top1_meters = {}
        loss_meters = {}

    model.eval()
    top1 = AverageMeter('Acc@1', '6.2:f')
    val_loss = AverageMeter('Loss', ':.4e')
    acc_list = []
    t_validate = time.time()
    with torch.no_grad():
        for density in density_list:
            t_start = time.time()
            top1.reset()
            val_loss.reset()

            Pruner(model, FLAGS.pruner, density)
            model = ComputePostBN.ComputeBN(model, post_loader)

            for input, target in loader:
                target = target.cuda(non_blocking=True)
                input = input.cuda(non_blocking=True)
                batch_size = target.size()[0]
                output = model(input)
                loss = torch.mean(criterion(output, target))
                acc1, _ = accuracy(output, target, topk=(1, 5))
                corr1, loss = acc1 * batch_size, loss * batch_size
                stats = torch.tensor([corr1, loss, batch_size]).cuda()
                dist.barrier()
                dist.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
                corr1, loss, batch_size = stats.tolist()
                acc1, loss = corr1 / batch_size, loss / batch_size
                top1.update(acc1, batch_size)
                val_loss.update(loss, batch_size)
            end_time = time.time()
            if dist.get_rank() == 0:
                top1_meters[f"VALID top1 x{density}"] = top1.avg
                loss_meters[f"VALID loss x{density}"] = val_loss.avg
                print(f"VALID {end_time - t_start:.1f}s | {epoch}/{FLAGS.num_epochs} {density} {top1.avg} {val_loss.avg}")
            acc_list.append(top1.avg)

    if dist.get_rank() == 0 and getattr(FLAGS, 'use_wandb', False):
        log = {}
        for metric in [top1_meters, loss_meters]:
            for name, meter in metric.items():
                log[name] = meter
                wandb.log(log,step=epoch)
        print(f"VALID COMPLETE {time.time() - t_validate:.1f}s")
    dist.barrier()
    return sum(acc_list) / len(acc_list)
       
def get_optimizer(model):
    if FLAGS.dataset == 'imagenet1k':
        model_params = []
        for params in model.parameters():
            ps = list(params.size())
            if len(ps) == 4 and ps[1] != 1:
                weight_decay = FLAGS.weight_decay
            elif len(ps) == 2:
                weight_decay = FLAGS.weight_decay
            else:
                weight_decay = 0
            
            item = {'params': params, 'weight_decay': weight_decay,
                    'lr': FLAGS.lr, 'momentum': FLAGS.momentum,
                    'nesterov': FLAGS.nesterov}
            model_params.append(item)
        optimizer = torch.optim.SGD(model_params)
    else:
        optimizer = torch.optim.SGD(model.parameters(), FLAGS.lr,
                                    momentum=FLAGS.momentum, nesterov=FLAGS.nesterov,
                                    weight_decay=FLAGS.weight_decay)
    return optimizer
    
model_profiling_hooks = []
name_space = 95
params_space = 15
macs_space = 15
def conv_module_name_filter(name):
    filters = {
        'kernel_size' : 'k',
        'stride' : 's',
        'padding': 'pad',
        'bias' : 'b',
        'groups' : 'g',
    }

    for k in filters:
        name = name.replace(k, filters[k])
    return name

def get_params(self):
    return np.sum([np.prod(list(w.size())) for w in self.parameters()])
def get_density(self):
    density = 1.0
    if self.mask is not None:
        mask = self.mask.cpu().numpy()
        density = np.count_nonzero(mask) / np.prod(mask.shape)
    return density

def module_profiling(self, input, output, verbose=None):
    ins = input[0].size()
    outs = output.size()
    t = type(self)
    if type(self) == nn.Conv2d:
        self.n_macs = (ins[1] * outs[1] *
                       self.kernel_size[0] * self.kernel_size[1] *
                       outs[2] * outs[3] // self.groups) * outs[0]
        self.n_params = get_params(self)
        self.name = conv_module_name_filter(self.__repr__())
    elif type(self) == DynamicGroupConv2d:
        density = get_density(self)
        self.n_macs = (ins[1] * outs[1] *
                       self.kernel_size[0] * self.kernel_size[1] *
                       outs[2] * outs[3] // self.groups) * outs[0] * density
        self.n_params = get_params(self) * density
        self.name = conv_module_name_filter(self.__repr__())
    elif isinstance(self, nn.Linear):
        self.n_macs = ins[1] * outs[1] * outs[0]
        self.n_params = get_params(self)
        self.name = self.__repr__()
    elif isinstance(self, nn.AdaptiveAvgPool2d):
        self.n_macs = ins[1] * ins[2] * ins[3] * ins[0]
        self.n_params = 0
        self.name = self.__repr__()
    else:
        self.n_macs = 0
        self.n_params = 0
        num_children = 0
        for m in self.children():
            self.n_macs += getattr(m, 'n_macs', 0)
            self.n_params += getattr(m, 'n_params', 0)
            num_children += 1
        ignore_zeros_t = [
            nn.BatchNorm2d, MyBatch, nn.Dropout2d, nn.Dropout, nn.Sequential,
            nn.ReLU6, nn.ReLU, nn.MaxPool2d,
            nn.modules.padding.ZeroPad2d, nn.modules.activation.Sigmoid,
        ]
        if (not getattr(self, 'ignore_model_profiling', False) and
                self.n_macs == 0 and
                t not in ignore_zeros_t):
                pass
        return

def add_profiling_hooks(m):
    global model_profiling_hooks
    model_profiling_hooks.append(
        m.register_forward_hook(lambda m, input, output: module_profiling(
            m, input, output)))

def remove_profiling_hooks():
    global model_profiling_hooks
    for h in model_profiling_hooks:
        h.remove()
    model_profiling_hooks = []

def model_profiling(model, height, width, batch=1, channel=3):
    model.eval()
    data = torch.rand(batch, channel, height, width)
    model.apply(lambda m: add_profiling_hooks(m))
    model(data)
    print(
        'Total'.ljust(name_space,' ') +
        '{}'.format(model.n_params//1e6).rjust(params_space, ' ') +
        '{}'.format(model.n_macs//1e6).rjust(macs_space, ' '))
    remove_profiling_hooks()
    
    
def main_worker(gpu, ngpus_per_node, args):
    global_rank = FLAGS.machine_rank * ngpus_per_node + gpu
    set_random_seed()
    dist.init_process_group(
        backend='nccl',
        init_method=FLAGS.dist_url, 
        world_size=FLAGS.world_size,
        rank=global_rank
    )
    if getattr(FLAGS, 'use_wandb', False) and gpu == 0:
        wandb.init(project='NEW LAMP ImageNet Training', name=FLAGS.log_dir, config=FLAGS.yaml(), resume='allow')
    dist.barrier()
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(gpu)
    model = get_model().cuda(gpu)
    model_wrapper = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[gpu], find_unused_parameters=False
    )

        
    #criterion = torch.nn.CrossEntropyLoss().cuda()
    #soft_criterion = KLLossSoft().cuda()
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    soft_criterion = CrossEntropyLossSoft(reduction='none')
    train_loader, val_loader, train_sampler  = get_dataset()

    if getattr(FLAGS, 'pretrained', False):
        checkpoint = torch.load(
            FLAGS.pretrained, map_location=lambda storage, loc: storage)
        if type(checkpoint) == dict and 'model' in checkpoint:
            checkpoint = checkpoint['model']
        new_keys = list(model_wrapper.state_dict().keys())
        old_keys = list(checkpoint.keys())
        new_keys = [key for key in new_keys if 'running' not in key]
        new_keys = [key for key in new_keys if 'tracked' not in key]
        old_keys = [key for key in old_keys if 'running' not in key]
        old_keys = [key for key in old_keys if 'tracked' not in key]
        
        new_checkpoint = {}
        for key_new, key_old in zip(new_keys, old_keys):
            new_checkpoint[key_new] = checkpoint[key_old]
        model_wrapper.load_state_dict(new_checkpoint, strict=False)
        print(f"Loaded model {FLAGS.pretrained}")
    
    optimizer = get_optimizer(model_wrapper)
    best_val = -1
    if os.path.exists(os.path.join(FLAGS.log_dir, 'latest_checkpoint.pt')):
        checkpoint = torch.load(os.path.join(FLAGS.log_dir, 'latest_checkpoint.pt'),
                                map_location=lambda storage, loc: storage)
        model_wrapper.load_state_dict(checkpoint['model'])
        #model_wrapper = torch.nn.parallel.DistributedDataParallel(
        #    model, device_ids=[gpu])
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['last_epoch']
        best_val = checkpoint['best_val']
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * FLAGS.num_epochs)
        lr_scheduler.last_epoch = last_epoch
        print(f"Loaded checkpoint {FLAGS.log_dir} at epoch {last_epoch}")
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * FLAGS.num_epochs)
        last_epoch = lr_scheduler.last_epoch
    
    if getattr(FLAGS, 'test_only', False):
        density_list = FLAGS.density_list
        validate(model_wrapper, val_loader, criterion, -1, train_loader, density_list)
        exit()

    print("Start Training")
    for epoch in range(last_epoch + 1, FLAGS.num_epochs):
        train_sampler.set_epoch(epoch)

        hist = _train_one_epoch(model_wrapper, train_loader, criterion, soft_criterion, optimizer, epoch, lr_scheduler)
        density_list = None

        if epoch % 10 == 1:
            density_list = FLAGS.density_list
        else:
            density_list = [max(FLAGS.density_list), min(FLAGS.density_list)]


        current_val = validate(model_wrapper, val_loader, criterion, epoch, train_loader, density_list)
        
        if current_val > best_val:
            best_val = current_val
            if dist.get_rank() == 0:
                if not os.path.exists(FLAGS.log_dir):
                    os.makedirs(FLAGS.log_dir)
                torch.save({
                    'model' : model_wrapper.state_dict(),
                    'best_val': current_val,
                },
                os.path.join(FLAGS.log_dir, 'best_model.pt'))
                print(f"New best Average Top-1 {best_val}")
        if dist.get_rank() == 0:
            torch.save({
                'model': model_wrapper.state_dict(),
                'optimizer': optimizer.state_dict(),
                'last_epoch': epoch,
                'best_val': best_val,
            },
            os.path.join(FLAGS.log_dir, f'latest_checkpoint_{epoch}.pt'))
            torch.save({
                'model': model_wrapper.state_dict(),
                'optimizer': optimizer.state_dict(),
                'last_epoch': epoch,
                'best_val': best_val,
            },
            os.path.join(FLAGS.log_dir, f'latest_checkpoint.pt'))
            np.savez(os.path.join(FLAGS.log_dir, f"gradient_history_{epoch}.npz"),**hist)
            hist = np.load(os.path.join(FLAGS.log_dir, f"gradient_history_{epoch}.npz"))

        
        if epoch % 5 == 1:
            if getattr(FLAGS, 'use_wandb', False) and dist.get_rank() == 0:
                with torch.no_grad():
                    for density in  FLAGS.density_list:
                        Pruner(model_wrapper, FLAGS.pruner, density)
                        record_sparsity(model_wrapper, density, epoch)

def main():
    mp.spawn(main_worker, nprocs=FLAGS.ngpus_per_node, args=(FLAGS.ngpus_per_node, FLAGS), join=True)
    if getattr(FLAGS, 'profiling', False):
        model = get_model()
        if os.path.exists(os.path.join(FLAGS.log_dir, 'latest_checkpoint.pt')):
            checkpoint = torch.load(os.path.join(FLAGS.log_dir, 'latest_checkpoint.pt'),
                                map_location=lambda storage, loc: storage)
            print(checkpoint)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            last_epoch = checkpoint['last_epoch']
            best_val = checkpoint['best_val']
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * FLAGS.num_epochs)
            lr_scheduler.last_epoch = last_epoch
            print(f"Loaded checkpoint {FLAGS.log_dir} at epoch {last_epoch}")
        else:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * FLAGS.num_epochs)
            last_epoch = lr_scheduler.last_epoch
        for density in FLAGS.density_list:
            Pruner(model, FLAGS.pruner, density)
            model_profiling(model, 224, 224, density)
if __name__ == '__main__':
    main()
