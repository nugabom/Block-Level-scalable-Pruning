import os
import torch
import torch.nn.functional as F
from utils.config import FLAGS
from torchvision import datasets, transforms
from utils.transforms import Lighting, InputList

def get_imagenet():
    if FLAGS.data_transforms == 'imagenet1k_basic':
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        crop_scale = 0.08
        jitter_param = 0.4
        lighting_param = 0.1
    elif FLAGS.data_transforms == 'iamgenet1k_mobile':
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        crop_scale = 0.25
        jitter_param = 0.4
        lighting_param = 0.1
    


    ## Multi - Scale Resolution Training

    if getattr(FLAGS, 'multi_scale', False): 
        print('multi_scale on')
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(FLAGS.image_size, scale=(crop_scale, 1.0)),
            transforms.ColorJitter(
                brightness=jitter_param,
                contrast=jitter_param,
                saturation=jitter_param),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            InputList(FLAGS.resolution_list),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(FLAGS.image_size, scale=(crop_scale, 1.0)),
            transforms.ColorJitter(
                brightness=jitter_param,
                contrast=jitter_param,
                saturation=jitter_param),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(FLAGS.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    
    train_set = datasets.ImageFolder(os.path.join(FLAGS.dataset_dir, 'train'), transform=train_transform)
    val_set = datasets.ImageFolder(os.path.join(FLAGS.dataset_dir, 'val'), transform=val_transform)
    
    if getattr(FLAGS, 'distributed', False):
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=FLAGS.batch_size, 
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=getattr(FLAGS, 'drop_last', False),
        pin_memory=True, 
        num_workers=FLAGS.data_loader_workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=FLAGS.batch_size, 
        shuffle=False,
        sampler=val_sampler,
        drop_last=getattr(FLAGS, 'drop_last', False),
        pin_memory=True, 
        num_workers=FLAGS.data_loader_workers,
    )

    return train_loader, val_loader, train_sampler


def get_cifar():
    if getattr(FLAGS, 'multi_scale', True):
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                        (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.507, 0.4865, 0.4409),
                                std=(0.2673, 0.2564, 0.2761)),
            InputList(FLAGS.resolution_list),

        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                        (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.507, 0.4865, 0.4409),
                                std=(0.2673, 0.2564, 0.2761)),

        ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.507, 0.4865, 0.4409),
                             std=(0.2673, 0.2564, 0.2761)),

    ])

    kwargs = {'num_workers': 1, 'pin_memory': True}
    assert (FLAGS.dataset == 'cifar10' or FLAGS.dataset == 'cifar100')
    train_loader = torch.utils.data.DataLoader(
        datasets.__dict__[FLAGS.dataset.upper()](FLAGS.dataset_dir,
                                                 train=True,
                                                 download=False,
                                                 transform=train_transform),
        batch_size=FLAGS.batch_size, shuffle=True, **kwargs)
    
    val_loader = torch.utils.data.DataLoader(
        datasets.__dict__[FLAGS.dataset.upper()](FLAGS.dataset_dir,
                                                 train=False,
                                                 download=False,
                                                 transform=val_transform),
        batch_size=FLAGS.batch_size, shuffle=True, **kwargs)

    return train_loader, val_loader


def get_dataset():
    if FLAGS.dataset == 'imagenet1k':
        return get_imagenet()
    elif 'cifar' in FLAGS.datset:
        return get_cifar()
    else:
        raise NotImplementedError("dataset not implemented.")


      


        
