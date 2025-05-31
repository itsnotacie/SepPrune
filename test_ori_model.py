import os
import random
from typing import Union
import soundfile as sf
import torch
import yaml
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from scipy.io import wavfile
import warnings
import copy
from ptflops import get_model_complexity_info
from look2hear.models.afrcnn_pruned import ConvNormAct
from look2hear.models.tdanet_v2 import ConvNorm, GlobLN
from look2hear.models.sudormrf import UConvBlock
# import torchaudio
warnings.filterwarnings("ignore")
import look2hear.models
import look2hear.datas
from look2hear.metrics import MetricsTracker
from look2hear.utils import tensors_to_device, RichProgressBarTheme, MyMetricsTextColumn, BatchesProcessedColumn
from torchsummary import summary

from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)



# 设置当前工作目录为文件所在的目录
os.chdir('/public/ly/other/Look2hear/')


parser = argparse.ArgumentParser()
parser.add_argument("--conf_dir",
                    default="configs/afrcnn12.yaml",
                    help="Full path to save best validation model")


compute_metrics = ["si_sdr", "sdr"]
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune




def main(config):
    metricscolumn = MyMetricsTextColumn(style=RichProgressBarTheme.metrics)
    progress = Progress(
        TextColumn("[bold blue]Testing", justify="right"),
        BarColumn(bar_width=None),
        "•",
        BatchesProcessedColumn(style=RichProgressBarTheme.batch_progress), 
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
        "•",
        metricscolumn
    )

    ### 加载retrain之后的权重
    # model_path = '/data/stan_2024/24_2024_spearate/Look2hear/experiments/checkpoint/TDANet_prune/best_model.pth'

    model =  getattr(look2hear.models, config["train_conf"]["audionet"]["audionet_name"])(
        # sample_rate=config["train_conf"]["datamodule"]["data_config"]["sample_rate"],
        **config["train_conf"]["audionet"]["audionet_config"],
    )

    def calculate_flops(model, input_size):
        macs, params = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=True)
        print(f"FLOPs: {macs}, Params: {params}")
        return macs, params

    # 示例用法
    input_size = (1,16000)
    original_flops, _ = calculate_flops(model, input_size)

    if config["train_conf"]["exp"]["exp_name"] == 'AFRCNN12_EchoSet':
        model_path = '/public/ly/other/Look2hear/experiments/checkpoint/AFRCNN12_EchoSet_ori/epoch=495.ckpt'
        model.load_state_dict({k.replace('audio_model.',''):v for k,v in torch.load(model_path)['state_dict'].items()})
    elif config["train_conf"]["exp"]["exp_name"] == 'AFRCNN16_EchoSet':
        model_path = '/public/ly/other/Look2hear/experiments/checkpoint/AFRCNN16_EchoSet_ori/epoch=326.ckpt'
        model.load_state_dict({k.replace('audio_model.',''):v for k,v in torch.load(model_path)['state_dict'].items()})
    elif config["train_conf"]["exp"]["exp_name"] == 'TDANet_EchoSet':
        model_path = '/public/ly/other/Look2hear/experiments/checkpoint/TDANet_EchoSet_ori/epoch=280.ckpt'
        model.load_state_dict({k.replace('audio_model.',''):v for k,v in torch.load(model_path)['state_dict'].items()})
        # sys.exit()
    elif config["train_conf"]["exp"]["exp_name"] == 'SuDORMRF_EchoSet':
        model_path = '/public/ly/other/Look2hear/experiments/checkpoint/SuDORMRF_EchoSet_ori/epoch=319.ckpt'
        model.load_state_dict({k.replace('audio_model.',''):v for k,v in torch.load(model_path)['state_dict'].items()})
        # sys.exit()
    elif config["train_conf"]["exp"]["exp_name"] == 'TDANet_Large':
        model_path = '/public/ly/other/Look2hear/experiments/checkpoint/TDANet_Large_ori/epoch=194.ckpt'
        model.load_state_dict({k.replace('audio_model.',''):v for k,v in torch.load(model_path)['state_dict'].items()})
    elif config["train_conf"]["exp"]["exp_name"] == 'AFRCNN16':
        model_path = '/public/ly/other/Look2hear/experiments/checkpoint/AFRCNN16_ori/epoch=161.ckpt'
        model.load_state_dict({k.replace('audio_model.',''):v for k,v in torch.load(model_path)['state_dict'].items()})
    elif config["train_conf"]["exp"]["exp_name"] == 'AFRCNN12_LibriMix':
        model_path = '/public/ly/other/Look2hear/experiments/checkpoint/AFRCNN12_LibriMix_ori/epoch=237.ckpt'
        model.load_state_dict({k.replace('audio_model.',''):v for k,v in torch.load(model_path)['state_dict'].items()})
    elif config["train_conf"]["exp"]["exp_name"] == 'TDANet_LibriMix':
        model_path = '/public/ly/other/Look2hear/experiments/checkpoint/TDANet_LibriMix_ori/epoch=117.ckpt'
        model.load_state_dict({k.replace('audio_model.',''):v for k,v in torch.load(model_path)['state_dict'].items()})
        # sys.exit()
    elif config["train_conf"]["exp"]["exp_name"] == 'SuDORMRF_LibriMix':
        model_path = '/public/ly/other/Look2hear/experiments/checkpoint/SuDORMRF_LibriMix_ori/epoch=286.ckpt'
        model.load_state_dict({k.replace('audio_model.',''):v for k,v in torch.load(model_path)['state_dict'].items()})
    # input_size = (1,16000)
    # original_flops, _ = calculate_flops(model, input_size)
    
    if config["train_conf"]["training"]["gpus"]:
        device = "cuda"
        model.to(device)
    # summary(model, input_size=(1, 200))
    model_device = next(model.parameters()).device
    print(config["train_conf"]["datamodule"]["data_config"])
    datamodule: object = getattr(look2hear.datas, config["train_conf"]["datamodule"]["data_name"])(
        **config["train_conf"]["datamodule"]["data_config"]
    )
    datamodule.setup()
    _, _ , test_set = datamodule.make_sets
   
    # Randomly choose the indexes of sentences to save.
    # ex_save_dir = os.path.join(config["train_conf"]["main_args"]["exp_dir"], "results/")
    ex_save_dir = os.path.join('./', "results/")
    os.makedirs(ex_save_dir, exist_ok=True)
    metrics = MetricsTracker(
        save_file=os.path.join(ex_save_dir, "metrics.csv"))
    torch.no_grad().__enter__()
    with progress:
        for idx in progress.track(range(len(test_set))):
            # Forward the network on the mixture.
            mix, sources, key = tensors_to_device(test_set[idx],
                                                    device=model_device)
            est_sources = model(mix[None])
            mix_np = mix
            sources_np = sources
            est_sources_np = est_sources.squeeze(0)
            metrics(mix=mix_np,
                    clean=sources_np,
                    estimate=est_sources_np,
                    key=key)
            # save_dir = "./TDANet"
            # # est_sources_np = normalize_tensor_wav(est_sources_np)
            # for i in range(est_sources_np.shape[0]):
            #     os.makedirs(os.path.join(save_dir, "s{}/".format(i + 1)), exist_ok=True)
                # torchaudio.save(os.path.join(save_dir, "s{}/".format(i + 1)) + key, est_sources_np[i].unsqueeze(0).cpu(), 16000)
            if idx % 50 == 0:
                metricscolumn.update(metrics.update())
    metrics.final()


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    # Load training config
    with open(args.conf_dir, "rb") as f:
        train_conf = yaml.safe_load(f)
    arg_dic["train_conf"] = train_conf
    # print(arg_dic)
    main(arg_dic)
