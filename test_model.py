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

    import pynvml
    def get_gpu_memory_usage():
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 0表示第一个 GPU
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_memory = info.used
        total_memory = info.total
        pynvml.nvmlShutdown()
        return used_memory, total_memory

    ### 加载retrain之后的权重
    # model_path = '/data/stan_2024/24_2024_spearate/Look2hear/experiments/checkpoint/TDANet_prune/best_model.pth'
    # used, total = get_gpu_mem
    # print(f"Total GPU Memory: {total / 1024**2:.2f} MB")

    model =  getattr(look2hear.models, config["train_conf"]["audionet"]["audionet_name"])(
        # sample_rate=config["train_conf"]["datamodule"]["data_config"]["sample_rate"],
        **config["train_conf"]["audionet"]["audionet_config"],
    )
    model.eval()

    def calculate_flops(model, input_size):
        macs, params = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=True)
        print(f"FLOPs: {macs}, Params: {params}")
        return macs, params

    # 示例用法
    # input_size = (1,16000)
    # original_flops, _ = calculate_flops(model, input_size)

    if config["train_conf"]["exp"]["exp_name"] == 'AFRCNN12':
        # model_path = '/public/ly/other/Look2hear/experiments/checkpoint/AFRCNN12_prune/epoch=94.ckpt'
        # model_path = '/public/ly/other/Look2hear/experiments/checkpoint/AFRCNN12_prune/epoch=123.ckpt'
        model_path = '/public/ly/other/Look2hear/experiments/checkpoint/AFRCNN12_prune_0.5/epoch=106.ckpt'
        # model_path = '/public/ly/other/Look2hear/experiments/checkpoint/AFRCNN12_prune_0.9/epoch=52.ckpt'


        # mask = torch.load('/public/ly/other/Look2hear/save/afrcnn12_LRS2_mask.npy')
        mask = torch.load('/public/ly/other/Look2hear/save/MASKAFRCNN12_mask_0.5.npy')
        # mask = torch.load('/public/ly/other/Look2hear/save/MASKAFRCNN12_mask_0.9.npy')
        mask1, mask2, mask3, mask4 = mask[0], mask[1], mask[2], mask[3]
        print(mask1)
        num1 = int(torch.sum(mask1))
        num2 = int(torch.sum(mask2))
        num3 = int(torch.sum(mask3))
        num4 = int(torch.sum(mask4))
        # print(num1,num2,num3,num4)
        total_mask = torch.cat((mask1, mask2, mask3, mask4))
        # print(total_mask.shape)
        
        model.sm.blocks.concat_layer[0] = ConvNormAct(1024, num1, 1)
        model.sm.blocks.concat_layer[1] = ConvNormAct(1536, num2, 1)
        model.sm.blocks.concat_layer[2] = ConvNormAct(1536, num3, 1)
        model.sm.blocks.concat_layer[3] = ConvNormAct(1024, num4, 1)
        model.sm.blocks.last_layer[0] = ConvNormAct(num1+num2+num3+num4, 512, 1)

        model.load_state_dict({k.replace('audio_model.',''):v for k,v in torch.load(model_path)['state_dict'].items()})
    elif config["train_conf"]["exp"]["exp_name"] == 'TDANet':
        # model_path = '/public/ly/other/Look2hear/experiments/checkpoint/TDANet_prune/epoch=25.ckpt'
        model_path = '/public/ly/other/Look2hear/experiments/checkpoint/TDANet_prune/epoch=52.ckpt'

        mask = torch.load('/public/ly/other/Look2hear/save/tdanet_LRS2_mask.npy')
        mask_fc1 = mask[0]
        num1 = int(torch.sum(mask_fc1))
        model.sm.unet.globalatt.mlp.fc1 = ConvNorm(512, num1, 1, bias=False)
        model.sm.unet.globalatt.mlp.fc1.norm = GlobLN(num1)
        model.sm.unet.globalatt.mlp.dwconv = nn.Conv1d(
            num1, num1, 5, 1, 2, bias=True, groups=num1
        )
        model.sm.unet.globalatt.mlp.fc2 = ConvNorm(num1, 512, 1, bias=False)
        model.load_state_dict({k.replace('audio_model.',''):v for k,v in torch.load(model_path)['state_dict'].items()})
        
        # sys.exit()
    elif config["train_conf"]["exp"]["exp_name"] == 'SuDORMRF':
        # model_path = '/public/ly/other/Look2hear/experiments/checkpoint/SuDORMRF_prune/epoch=62.ckpt'
        # model_path = '/public/ly/other/Look2hear/experiments/checkpoint/SuDORMRF_prune/epoch=153.ckpt'
        model_path = '/public/ly/other/Look2hear/experiments/checkpoint/SuDORMRF_prune/epoch=161.ckpt'

        mask = torch.load('/public/ly/other/Look2hear/save/sudormrf_LRS2_mask.npy')
        num_list = []
        for i in range(len(mask)):
            num1 = int(torch.sum(mask[i]))
            num_list.append(num1)
        
        for i in range(16):
            model.sm[i] = UConvBlock(out_channels=128,
                        in_channels=num_list[i],
                        upsampling_depth=5)
        model.load_state_dict({k.replace('audio_model.',''):v for k,v in torch.load(model_path)['state_dict'].items()})
        # del mask
    elif config["train_conf"]["exp"]["exp_name"] == 'AFRCNN12_random':  # same mask but train from scratch
        print('AFRCNN same mask but train from scratch')
        model_path = '/public/ly/other/Look2hear/experiments/checkpoint/AFRCNN12_random_prune/epoch=30.ckpt'

        mask = torch.load('/public/ly/other/Look2hear/save/afrcnn12_LRS2_mask.npy')
        mask1, mask2, mask3, mask4 = mask[0], mask[1], mask[2], mask[3]
        print(mask1)
        num1 = int(torch.sum(mask1))
        num2 = int(torch.sum(mask2))
        num3 = int(torch.sum(mask3))
        num4 = int(torch.sum(mask4))
        # print(num1,num2,num3,num4)
        total_mask = torch.cat((mask1, mask2, mask3, mask4))
        # print(total_mask.shape)
        
        model.sm.blocks.concat_layer[0] = ConvNormAct(1024, num1, 1)
        model.sm.blocks.concat_layer[1] = ConvNormAct(1536, num2, 1)
        model.sm.blocks.concat_layer[2] = ConvNormAct(1536, num3, 1)
        model.sm.blocks.concat_layer[3] = ConvNormAct(1024, num4, 1)
        model.sm.blocks.last_layer[0] = ConvNormAct(num1+num2+num3+num4, 512, 1)

        model.load_state_dict({k.replace('audio_model.',''):v for k,v in torch.load(model_path)['state_dict'].items()})
    elif config["train_conf"]["exp"]["exp_name"] == 'TDANet_random':  # same mask but train from scratch
        # model_path = '/public/ly/other/Look2hear/experiments/checkpoint/TDANet_prune/epoch=25.ckpt'
        model_path = '/public/ly/other/Look2hear/experiments/checkpoint/TDANet_random_prune/epoch=35.ckpt'

        mask = torch.load('/public/ly/other/Look2hear/save/tdanet_LRS2_mask.npy')
        mask_fc1 = mask[0]
        num1 = int(torch.sum(mask_fc1))
        model.sm.unet.globalatt.mlp.fc1 = ConvNorm(512, num1, 1, bias=False)
        model.sm.unet.globalatt.mlp.fc1.norm = GlobLN(num1)
        model.sm.unet.globalatt.mlp.dwconv = nn.Conv1d(
            num1, num1, 5, 1, 2, bias=True, groups=num1
        )
        model.sm.unet.globalatt.mlp.fc2 = ConvNorm(num1, 512, 1, bias=False)
        model.load_state_dict({k.replace('audio_model.',''):v for k,v in torch.load(model_path)['state_dict'].items()})
        # sys.exit()
    elif config["train_conf"]["exp"]["exp_name"] == 'SuDORMRF_random':  # same mask but train from scratch
        # model_path = '/public/ly/other/Look2hear/experiments/checkpoint/SuDORMRF_prune/epoch=62.ckpt'
        print(1111111)
        model_path = '/public/ly/other/Look2hear/experiments/checkpoint/SuDORMRF_random_prune/epoch=7.ckpt'

        mask = torch.load('/public/ly/other/Look2hear/save/sudormrf_LRS2_mask.npy')
        num_list = []
        for i in range(len(mask)):
            num1 = int(torch.sum(mask[i]))
            num_list.append(num1)
        
        for i in range(16):
            model.sm[i] = UConvBlock(out_channels=128,
                        in_channels=num_list[i],
                        upsampling_depth=5)
        model.load_state_dict({k.replace('audio_model.',''):v for k,v in torch.load(model_path)['state_dict'].items()})
        print('same mask but train from scratch')

    elif config["train_conf"]["exp"]["exp_name"] == 'AFRCNN12_EchoSet':  
        model_path = '/public/ly/other/Look2hear/experiments/checkpoint/AFRCNN12_EchoSet_prune/epoch=341.ckpt'

        mask = torch.load('/public/ly/other/Look2hear/save/MASKAFRCNN12_EchoSet_mask.npy')
        mask1, mask2, mask3, mask4 = mask[0], mask[1], mask[2], mask[3]
        print(mask1)
        num1 = int(torch.sum(mask1))
        num2 = int(torch.sum(mask2))
        num3 = int(torch.sum(mask3))
        num4 = int(torch.sum(mask4))
        # print(num1,num2,num3,num4)
        total_mask = torch.cat((mask1, mask2, mask3, mask4))
        # print(total_mask.shape)
        
        model.sm.blocks.concat_layer[0] = ConvNormAct(1024, num1, 1)
        model.sm.blocks.concat_layer[1] = ConvNormAct(1536, num2, 1)
        model.sm.blocks.concat_layer[2] = ConvNormAct(1536, num3, 1)
        model.sm.blocks.concat_layer[3] = ConvNormAct(1024, num4, 1)
        model.sm.blocks.last_layer[0] = ConvNormAct(num1+num2+num3+num4, 512, 1)

        model.load_state_dict({k.replace('audio_model.',''):v for k,v in torch.load(model_path)['state_dict'].items()})
    elif config["train_conf"]["exp"]["exp_name"] == 'TDANet_EchoSet':  # same mask but train from scratch
        # model_path = '/public/ly/other/Look2hear/experiments/checkpoint/TDANet_prune/epoch=25.ckpt'
        model_path = '/public/ly/other/Look2hear/experiments/checkpoint/TDANet_EchoSet_prune/epoch=207.ckpt'

        mask = torch.load('/public/ly/other/Look2hear/save/MaskTDANet_EchoSet_mask.npy')
        mask_fc1 = mask[0]
        num1 = int(torch.sum(mask_fc1))
        model.sm.unet.globalatt.mlp.fc1 = ConvNorm(512, num1, 1, bias=False)
        model.sm.unet.globalatt.mlp.fc1.norm = GlobLN(num1)
        model.sm.unet.globalatt.mlp.dwconv = nn.Conv1d(
            num1, num1, 5, 1, 2, bias=True, groups=num1
        )
        model.sm.unet.globalatt.mlp.fc2 = ConvNorm(num1, 512, 1, bias=False)
        model.load_state_dict({k.replace('audio_model.',''):v for k,v in torch.load(model_path)['state_dict'].items()})
        # sys.exit()
    elif config["train_conf"]["exp"]["exp_name"] == 'SuDORMRF_EchoSet':  
        print('SuDORMRF_EchoSet')
        # model_path = '/public/ly/other/Look2hear/experiments/checkpoint/SuDORMRF_prune/epoch=62.ckpt'
        model_path = '/public/ly/other/Look2hear/experiments/checkpoint/SuDORMRF_EchoSet_prune/epoch=197.ckpt'

        mask = torch.load('/public/ly/other/Look2hear/save/MASKSuDORMRF_EchoSet_mask.npy')
        num_list = []
        for i in range(len(mask)):
            num1 = int(torch.sum(mask[i]))
            num_list.append(num1)
        
        for i in range(16):
            model.sm[i] = UConvBlock(out_channels=128,
                        in_channels=num_list[i],
                        upsampling_depth=5)
        model.load_state_dict({k.replace('audio_model.',''):v for k,v in torch.load(model_path)['state_dict'].items()})
    elif config["train_conf"]["exp"]["exp_name"] == 'TDANet_LibriMix':  # same mask but train from scratch
        # model_path = '/public/ly/other/Look2hear/experiments/checkpoint/TDANet_prune/epoch=25.ckpt'
        model_path = '/public/ly/other/Look2hear/experiments/checkpoint/TDANet_LibriMix_prune/epoch=128.ckpt'

        mask = torch.load('/public/ly/other/Look2hear/save/MaskTDANet_LibriMix_mask.npy')
        mask_fc1 = mask[0]
        num1 = int(torch.sum(mask_fc1))
        model.sm.unet.globalatt.mlp.fc1 = ConvNorm(512, num1, 1, bias=False)
        model.sm.unet.globalatt.mlp.fc1.norm = GlobLN(num1)
        model.sm.unet.globalatt.mlp.dwconv = nn.Conv1d(
            num1, num1, 5, 1, 2, bias=True, groups=num1
        )
        model.sm.unet.globalatt.mlp.fc2 = ConvNorm(num1, 512, 1, bias=False)
        model.load_state_dict({k.replace('audio_model.',''):v for k,v in torch.load(model_path)['state_dict'].items()})
    elif config["train_conf"]["exp"]["exp_name"] == 'SuDORMRF_LibriMix':  
        print('SuDORMRF_LibriMix')
        # model_path = '/public/ly/other/Look2hear/experiments/checkpoint/SuDORMRF_prune/epoch=62.ckpt'
        model_path = '/public/ly/other/Look2hear/experiments/checkpoint/SuDORMRF_LibriMix_prune/epoch=400.ckpt'

        mask = torch.load('/public/ly/other/Look2hear/save/MASKSuDORMRF_LibriMix_mask.npy')
        num_list = []
        for i in range(len(mask)):
            num1 = int(torch.sum(mask[i]))
            num_list.append(num1)
        
        for i in range(16):
            model.sm[i] = UConvBlock(out_channels=128,
                        in_channels=num_list[i],
                        upsampling_depth=5)
        model.load_state_dict({k.replace('audio_model.',''):v for k,v in torch.load(model_path)['state_dict'].items()})
    elif config["train_conf"]["exp"]["exp_name"] == 'AFRCNN12_LibriMix':  
        model_path = '/public/ly/other/Look2hear/experiments/checkpoint/AFRCNN12_LibriMix_prune_0.8/epoch=325.ckpt'

        mask = torch.load('/public/ly/other/Look2hear/save/MASKAFRCNN12_LibriMix_mask.npy')
        mask1, mask2, mask3, mask4 = mask[0], mask[1], mask[2], mask[3]
        print(mask1)
        num1 = int(torch.sum(mask1))
        num2 = int(torch.sum(mask2))
        num3 = int(torch.sum(mask3))
        num4 = int(torch.sum(mask4))
        # print(num1,num2,num3,num4)
        total_mask = torch.cat((mask1, mask2, mask3, mask4))
        # print(total_mask.shape)
        
        model.sm.blocks.concat_layer[0] = ConvNormAct(1024, num1, 1)
        model.sm.blocks.concat_layer[1] = ConvNormAct(1536, num2, 1)
        model.sm.blocks.concat_layer[2] = ConvNormAct(1536, num3, 1)
        model.sm.blocks.concat_layer[3] = ConvNormAct(1024, num4, 1)
        model.sm.blocks.last_layer[0] = ConvNormAct(num1+num2+num3+num4, 512, 1)

        model.load_state_dict({k.replace('audio_model.',''):v for k,v in torch.load(model_path)['state_dict'].items()})

    def get_gpu_memory_usage(model, input_data):
        # 将模型和输入数据移动到 GPU
        model = model.cuda()
        input_data = input_data.cuda()

        # 清空缓存
        torch.cuda.empty_cache()

        # 获取模型初始显存占用
        initial_memory = torch.cuda.memory_allocated()

        # 前向传播
        with torch.no_grad():
            output = model(input_data)

        # 获取前向传播后显存占用
        final_memory = torch.cuda.memory_allocated()

        # 计算模型显存占用
        model_memory = final_memory - initial_memory

        return model_memory

    input_size = (1,16000)
    original_flops, _ = calculate_flops(model, input_size)
    
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
    import time
    start_time = time.time()
    # used, total = get_gpu_memory_usage()
    # print(f"Used GPU Memory: {used / 1024**2:.2f} MB")
    input_data = torch.randn((1,1,16000))
    memory_usage = get_gpu_memory_usage(model, input_data)
    print(f"Model GPU Memory Usage: {memory_usage / 1024**2:.2f} MB")
    def attach_mem_hooks(model):
        """
        为 model 的每个子模块注册 hook，用于记录推理时的显存占用。
        返回一个 dict: {module_name: [mem_before, mem_after, peak_mem]}。
        """
        stats = {}

        def pre_hook(module, input):
            # 推理进入该模块前，重置峰值并记录当前已用显存
            torch.cuda.reset_peak_memory_stats()
            stats[module_name(module)]['mem_before'] = torch.cuda.memory_allocated()

        def post_hook(module, input, output):
            # 模块执行后记录峰值显存和当前显存
            stats[module_name(module)]['mem_after'] = torch.cuda.memory_allocated()
            stats[module_name(module)]['peak_mem'] = torch.cuda.max_memory_allocated()

        for name, module in model.named_modules():
            # 只对有参数或常用的层注册（可根据需要过滤）
            if len(list(module.parameters())) > 0:
                stats[name] = {'mem_before': 0, 'mem_after': 0, 'peak_mem': 0}
                module.register_forward_pre_hook(pre_hook)
                module.register_forward_hook(post_hook)

        return stats

    def module_name(module):
        """
        简单辅助，把 module 转成可用作 stats 字典 key 的字符串。
        """
        for name, mod in model.named_modules():
            if mod is module:
                return name
        return str(module)

    # 附加钩子并准备统计容器
    mem_stats = attach_mem_hooks(model)

    # 准备一条示例输入（batch_size、channels、time_length 根据实际情况调整）
    dummy_input = torch.randn(1, 1, 16000).cuda()

    # 执行一次推理
    with torch.no_grad():
        _ = model(dummy_input)

    # 推理结束后，打印每个子模块的显存信息
    print(f"{'Module':40s} | {'Before (MB)':>12s} | {'After (MB)':>11s} | {'Peak (MB)':>9s}")
    print("-" * 100)
    total_before = 0
    total_after = 0
    total_peak = 0
    for name, stat in mem_stats.items():
        b = stat['mem_before'] / 1024**2
        total_before += b
        a = stat['mem_after']  / 1024**2
        p = stat['peak_mem']   / 1024**2
        total_after += a
        total_peak += p
        print(f"{name:40s} | {b:12.2f} | {a:11.2f} | {p:9.2f}")
    
    print(total_before, total_after, total_peak)

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
    end_time = time.time()
    print('cost time: {}'.format(end_time-start_time))

if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    # Load training config
    with open(args.conf_dir, "rb") as f:
        train_conf = yaml.safe_load(f)
    arg_dic["train_conf"] = train_conf
    # print(arg_dic)
    main(arg_dic)
