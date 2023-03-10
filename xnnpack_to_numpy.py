import importlib
import os
import logging

import torch
import numpy as np
import warnings
from math import isclose
from models.new_group_level_ops import *
from models.mobilenet_v2 import Pytorch_MobileNetV2

from torch.autograd import Variable
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.layers import *
from keras.applications.mobilenet_v2 import MobileNetV2
import tensorflow_model_optimization as tfmot

import onnx
from onnx2keras import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last as ctfl

def get_model():
    model_lib = importlib.import_module(FLAGS.model)
    model = model_lib.Model(FLAGS.num_classes, FLAGS.image_size)

    return model

def hard_prune(model, pruner, density):
    if density == 1.0:
        print(f"hard_prune: no prune the Full model")
        return

    debug = {}
    Pruner(model, FLAGS.pruner, density)
    for conv in model.modules():
        if isinstance(conv, DynamicGroupConv2d):
            mask = conv.mask.cpu().detach().numpy().astype(np.float32)
            weight = conv.weight.cpu().detach().numpy().astype(np.float32)
            weight = np.ones_like(weight) + np.absolute(weight)
            weight = weight.astype(np.float32)
            conv.weight.data = torch.from_numpy(weight * mask).to(conv.weight.device)
            debug[f"{conv.name}"] = np.count_nonzero(mask)/np.prod(mask.shape)
    return debug

def create_source_file(main_folder, layer_id, density):
    layer_folder_name = os.path.join(main_folder, f"prunable_layer_{layer_id}")
    if not os.path.exists(layer_folder_name):
        os.makedirs(layer_folder_name)
    return os.path.join(layer_folder_name, f"{density}_mask.npz")

if __name__ == "__main__":
    density = float(os.environ["density"])
    logging.basicConfig(level=logging.DEBUG)
    model = get_model()

    if os.path.exists(os.path.join(FLAGS.log_dir, 'latest_checkpoint.pt')):
        checkpoint = torch.load(os.path.join(FLAGS.log_dir, 'latest_checkpoint.pt'), map_location=lambda storage, loc: storage)

        model.load_state_dict(checkpoint['model'])
        last_epoch = checkpoint['best_val']
        best_val = checkpoint['best_val']
        
        print(f"------------------------------------")
        print(f"Loaded check point {FLAGS.log_dir}")
        print(f"best val {best_val}")
    else:
        warnings.warn(f"File is not exisit {FLAGS.log_dir}")
        assert 1==0, f"File is not exisit {FLAGS.log_dir}"

    debug = hard_prune(model, FLAGS.pruner, density)
    for key, value in debug.items():
        print(f"{key} = {value}")
    
    temp = [layer.mask.cpu().detach() for layer in model.modules() if isinstance(layer, DynamicGroupConv2d)]
    layer_sparse_fmt = []
    for mask in temp:
        o, i, h, w = mask.size()
        mask = mask.reshape(o // 2, 2, i).mean(dim=1).view(-1).numpy()
        cnt = np.count_nonzero(mask) * 2
        value = np.random.uniform(low=1, high=200, size=mask.shape)
        mask *= value
        layer_sparse_fmt.append((mask, cnt))

    
    base_folder = f"real_layer_mask"
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    model_folder_name = f"xnnpack_{FLAGS.pruner}__mobilenetV2"
    main_folder = os.path.join(base_folder, model_folder_name)
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)

    for layer_id, (b_matrix, num_nonzeroes) in enumerate(layer_sparse_fmt):
        np.savez(create_source_file(main_folder, layer_id, density), b_matrix)
       


    










