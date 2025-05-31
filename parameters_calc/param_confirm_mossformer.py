import sys
sys.path.append('/public/ly/other/Look2hear')
from ptflops import get_model_complexity_info
from look2hear import models
import argparse
import yaml
from look2hear.utils import tensors_to_device
import os
import torch
import torch.nn as nn
import look2hear.models

parser = argparse.ArgumentParser()
parser.add_argument("--conf_dir",
                    default='configs/mossformer.yaml', #"Experiments/checkpoint/TDANet/conf.yml"
                    help="Full path to save best validation model")
args = parser.parse_args()
config = dict(vars(args))

# Load training config
with open(args.conf_dir, "rb") as f:
    train_conf = yaml.safe_load(f)
config = train_conf

# config["train_conf"]["main_args"]["exp_dir"] = os.path.join(
#     os.getcwd(), "Experiments", "checkpoint", config["train_conf"]["exp"]["exp_name"]
# )

# model_path = os.path.join(config["train_conf"]["main_args"]["exp_dir"], "best_model.pth")
model = getattr(look2hear.models, config["audionet"]["audionet_name"])(
        # sample_rate=config["datamodule"]["data_config"]["sample_rate"],
        **config["audionet"]["audionet_config"],
    )
# .from_pretrain(model_path,**config["train_conf"]["audionet"]["audionet_config"]).cuda()

model.eval()

### 剪枝操作#############
def prune_layer_weights(model, param_paths):
    for name, param in model.named_parameters():
        if name in param_paths:
            print(name,' pruned!')
            param.data.zero_()
    return model

##剪枝sm.unet.loc_glo_fus.4
param_paths = [
    'sm.unet.loc_glo_fus.4.local_embedding.conv.weight',
    'sm.unet.loc_glo_fus.4.local_embedding.norm.gamma',
    'sm.unet.loc_glo_fus.4.local_embedding.norm.beta',
    'sm.unet.loc_glo_fus.4.global_embedding.conv.weight',
    'sm.unet.loc_glo_fus.4.global_embedding.norm.gamma',
    'sm.unet.loc_glo_fus.4.global_embedding.norm.beta',
    'sm.unet.loc_glo_fus.4.global_act.conv.weight',
    'sm.unet.loc_glo_fus.4.global_act.norm.gamma',
    'sm.unet.loc_glo_fus.4.global_act.norm.beta'
]
# model = prune_layer_weights(model, param_paths)

# def remove_layers_by_name(model, param_paths):
#     """按路径删除指定层"""
#     for path in param_paths:
#         # 分离模块路径
#         module_path = path.rsplit('.', 1)[0]
#         param_name = path.rsplit('.', 1)[-1]

#         # 获取父模块
#         parent_module = model
#         for part in module_path.split('.'):
#             parent_module = getattr(parent_module, part)

#         # 删除指定参数
#         if hasattr(parent_module, param_name):
#             print(f"✅ 删除 {path}")
#             delattr(parent_module, param_name)

#     return model
# model = remove_layers_by_name(model, param_paths)

# #########################





# 计算模型原始FLOPs
def calculate_flops(model, input_size):
    macs, params = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=True)
    print(f"FLOPs: {macs}, Params: {params}")
    return macs, params



# 示例用法
input_size = (1,48000)
original_flops, _ = calculate_flops(model, input_size)

# import torch.nn.utils.prune as prune
# # 对特定层的权重做结构化剪枝
# for name, module in model.named_modules():
#     if 'sm.unet.loc_glo_fus.4' in name or 'sm.unet.loc_glo_fus.2' in name or 'sm.unet.spp_dw.4' in name :
#         if hasattr(module, "weight"):
#             prune.ln_structured(module, name='weight', amount=0.5, n=2, dim=0)

# # 移除剪枝的掩码（让剪枝后的模型更干净）
# for name, module in model.named_modules():
#     if 'sm.unet.loc_glo_fus.4' in name or 'sm.unet.loc_glo_fus.2' in name or 'sm.unet.spp_dw.4' in name:
#         if hasattr(module, "weight"):
#             prune.remove(module, 'weight')


            
# original_flops, _ = calculate_flops(model, input_size)



# flops_dynamic, params_dynamic = get_model_complexity_info(
# model,
# (1,48000),
# as_strings=True,
# print_per_layer_stat=True
# )
# print("Dynamic Model (k=64):")
# print(f"  FLOPs: {flops_dynamic}")
# print(f"  Params: {params_dynamic}")