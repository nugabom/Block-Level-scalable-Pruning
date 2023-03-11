# Scable
Block-Level Scalable Pruning for IoT devices (e.g. MCU, mobile-device)
speed up and better accuracy channel-level scalable Pruning (USNet [ICCV' 19] https://arxiv.org/pdf/1903.05134.pdf)

## Pruning method
to apply to various pruning method, adding pruning method "models/new_group_level_ops.py"

## Training Code
for single GPU training
train.py (e.g. python3 train.py app:apps/mobilenet_v1_config.yml)

for Multi-GPU training
new_train.py (e.g. CUDA_VISIBLE_CUDA=NUMBER_OF_GPUS new_train.py app:apps/mobilenet_v1_config.yml)


## XNNPACK covert
'xnnpack_experiment.py' will convert model file to xnnpack end-to-end code  
