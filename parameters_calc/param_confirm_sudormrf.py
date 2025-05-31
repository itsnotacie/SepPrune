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
                    default='configs/sudormrf.yaml', 
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





# 计算模型原始FLOPs
def calculate_flops(model, input_size):
    macs, params = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=True)
    print(f"FLOPs: {macs}, Params: {params}")
    return macs, params



# 示例用法
input_size = (1,16000)
original_flops, _ = calculate_flops(model, input_size)
