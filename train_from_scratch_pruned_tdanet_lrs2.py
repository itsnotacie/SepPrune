import os
import sys
import torch
from torch import Tensor
import argparse
import copy
import json
from ptflops import get_model_complexity_info
import random
import torch.nn as nn
import look2hear.datas
import look2hear.models
import look2hear.system
import look2hear.losses
import look2hear.metrics
import look2hear.utils
from look2hear.system import make_optimizer
from dataclasses import dataclass
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import *
from rich.console import Console
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from rich import print, reconfigure
from collections.abc import MutableMapping
from look2hear.utils import print_only, MyRichProgressBar, RichProgressBarTheme
import torch.nn.utils.prune as prune
from torchsummary import summary
from look2hear.models.tdanet_v2 import ConvNorm, GlobLN

import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--conf_dir",
    default="configs/tdanet_prune.yml",
    help="Full path to save best validation model",
)

def main(config):
    print_only(
        "Instantiating datamodule <{}>".format(config["datamodule"]["data_name"])
    )
    datamodule: object = getattr(look2hear.datas, config["datamodule"]["data_name"])(
        **config["datamodule"]["data_config"]
    )
    datamodule.setup()

    train_loader, val_loader, test_loader = datamodule.make_loader
    
    # Define model and optimizer
####################################################################################################################
##### 随机初始化，从scrach训练
    print_only(
        "Instantiating AudioNet <{}>".format(config["audionet"]["audionet_name"])
    )
    model = getattr(look2hear.models, config["audionet"]["audionet_name"])(
        # sample_rate=config["datamodule"]["data_config"]["sample_rate"],
        **config["audionet"]["audionet_config"],
    )

    if config["exp"]["exp_name"] == 'TDANet':
        print('using TDANet')
        model_path = '/public/ly/other/Look2hear/look2hear/exp/tdanet/tdanet_LRS2_bak_best_model.pth'
    elif config["exp"]["exp_name"] == 'TDANet_Large':
        print('using TDANet_Large, not finished')
        sys.exit()

    def calculate_flops(model, input_size):
        macs, params = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=True)
        print(f"FLOPs: {macs}, Params: {params}")

    # 示例用法
    # input_size = (1,16000)
    # calculate_flops(model, input_size)
    # print(model)
    mask = torch.load('/public/ly/other/Look2hear/save/tdanet_LRS2_mask.npy')
    mask_fc1 = mask[0]
    num1 = int(torch.sum(mask_fc1))
    print(num1)

    pruned_model = copy.deepcopy(model)
    pruned_model.sm.unet.globalatt.mlp.fc1 = ConvNorm(512, num1, 1, bias=False)
    pruned_model.sm.unet.globalatt.mlp.fc1.norm = GlobLN(num1)
    pruned_model.sm.unet.globalatt.mlp.dwconv = nn.Conv1d(
            num1, num1, 5, 1, 2, bias=True, groups=num1
        )
    pruned_model.sm.unet.globalatt.mlp.fc2 = ConvNorm(num1, 512, 1, bias=False)
####################################################################################################################    
    
    
    # Remove blocks if needed (consistent across processes)
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            block_indices_to_remove = random.sample(range(len(pruned_model.sm.blocks.spp_dw)), k=2)
        else:
            block_indices_to_remove = None
        block_indices_to_remove = torch.tensor(block_indices_to_remove)
        torch.distributed.broadcast(block_indices_to_remove, src=0)
        pruned_model.sm.blocks.remove_blocks(block_indices_to_remove=block_indices_to_remove.tolist())


    # import pdb; pdb.set_trace()
    print_only("Instantiating Optimizer <{}>".format(config["optimizer"]["optim_name"]))
    optimizer = make_optimizer(pruned_model.parameters(), **config["optimizer"])

    # Define scheduler
    scheduler = None
    if config["scheduler"]["sche_name"]:
        print_only(
            "Instantiating Scheduler <{}>".format(config["scheduler"]["sche_name"])
        )
        if config["scheduler"]["sche_name"] != "DPTNetScheduler":
            scheduler = getattr(torch.optim.lr_scheduler, config["scheduler"]["sche_name"])(
                optimizer=optimizer, **config["scheduler"]["sche_config"]
            )
        else:
            scheduler = {
                "scheduler": getattr(look2hear.system.schedulers, config["scheduler"]["sche_name"])(
                    optimizer, len(train_loader) // config["datamodule"]["data_config"]["batch_size"], 64
                ),
                "interval": "step",
            }

    # Just after instantiating, save the args. Easy loading in the future.
    config["main_args"]["exp_dir"] = os.path.join(
        os.getcwd(), "experiments", "checkpoint", config["exp"]["exp_name"]+'_prune'
    )
    exp_dir = config["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(config, outfile)

    # Define Loss function.
    print_only(
        "Instantiating Loss, Train <{}>, Val <{}>".format(
            config["loss"]["train"]["sdr_type"], config["loss"]["val"]["sdr_type"]
        )
    )
    loss_func = {
        "train": getattr(look2hear.losses, config["loss"]["train"]["loss_func"])(
            getattr(look2hear.losses, config["loss"]["train"]["sdr_type"]),
            **config["loss"]["train"]["config"],
        ),
        "val": getattr(look2hear.losses, config["loss"]["val"]["loss_func"])(
            getattr(look2hear.losses, config["loss"]["val"]["sdr_type"]),
            **config["loss"]["val"]["config"],
        ),
    }

    print_only("Instantiating System <{}>".format(config["training"]["system"]))
    system = getattr(look2hear.system, config["training"]["system"])(
        audio_model=pruned_model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scheduler=scheduler,
        config=config,
    )

    # Define callbacks
    print_only("Instantiating ModelCheckpoint")
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir)
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        filename="{epoch}",
        monitor="val_loss/dataloader_idx_0",
        mode="min",
        save_top_k=5,
        verbose=True,
        save_last=True,
    )
    callbacks.append(checkpoint)

    if config["training"]["early_stop"]:
        print_only("Instantiating EarlyStopping")
        callbacks.append(EarlyStopping(**config["training"]["early_stop"]))
    callbacks.append(MyRichProgressBar(theme=RichProgressBarTheme()))

    # Don't ask GPU if they are not available.
    gpus = config["training"]["gpus"] if torch.cuda.is_available() else None
    distributed_backend = "cuda" if torch.cuda.is_available() else None

    # default logger used by trainer
    logger_dir = os.path.join(os.getcwd(), "Experiments", "tensorboard_logs")
    os.makedirs(os.path.join(logger_dir, config["exp"]["exp_name"]), exist_ok=True)
    # comet_logger = TensorBoardLogger(logger_dir, name=config["exp"]["exp_name"])
    comet_logger = WandbLogger(
            name=config["exp"]["exp_name"], 
            save_dir=os.path.join(logger_dir, config["exp"]["exp_name"]), 
            project="Look2hear",
            # offline=True
    )

    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        devices=gpus,
        accelerator=distributed_backend,
        strategy=DDPStrategy(find_unused_parameters=True),
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=5.0,
        logger=comet_logger,
        sync_batchnorm=True,
        # num_sanity_val_steps=0,
        # sync_batchnorm=True,
        # fast_dev_run=True,
    )
    trainer.fit(system)
    print_only("Finished Training")
    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.audio_model.serialize()
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    import yaml
    from pprint import pprint
    from look2hear.utils.parser_utils import (
        prepare_parser_from_dict,
        parse_args_as_dict,
    )

    args = parser.parse_args()
    with open(args.conf_dir) as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)

    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    # pprint(arg_dic)
    main(arg_dic)
