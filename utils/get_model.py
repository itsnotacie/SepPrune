import sys
import heapq
import copy
import numpy as np
from torchvision.models import resnet50

from model.ResNet import ResNet50_13
from model.VGG_cifar import *
import torch
from model.samll_resnet import *
from utils.get_params import get_cresnet_layer_params, load_cresnet_layer_params, load_Iresnet_layer_params, \
    get_Iresnet_layer_params, load_vgg_layer_params


def get_model(args):
    # Note that you can train your own models using train.py
    print(f"=> Getting {args.arch}")
    if args.arch == 'ResNet50':
        model = resnet50(pretrained=False)
        if args.pretrained:
            model = resnet50(pretrained=True)
    elif args.arch == 'cvgg16_bn':
        model = cvgg16_bn(num_classes=args.num_classes, batch_norm=True)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg16_bn/cifar10/scores.pt', map_location='cuda:%d' % args.gpu)
                model.load_state_dict(ckpt)
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg16_bn/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
                model.load_state_dict(ckpt)
    elif args.arch == 'resnet56':
        model = resnet56(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet56.th', map_location='cuda:%d' % args.gpu)
                ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
                model.load_state_dict(ckpt)
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet56/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
                model.load_state_dict(ckpt)
    elif args.arch == 'resnet56_KD_15_c100':
        model = resnet56_KD_15_c100(num_classes=100)
        if args.finetune:
            orginal_model = resnet56(num_classes=100)
            ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet56/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
            orginal_model.load_state_dict(ckpt)
            orginal_state_list = get_cresnet_layer_params(orginal_model)
            remain_list = [[0, 1, 2, 3, 5], [2, 3, 4], [1, 2, 3, 4, 6, 8]]
            model = load_cresnet_layer_params(orginal_state_list, model, remain_list, num_of_block=9)
            print('Load pretrained weights from the original model')
        if args.pretrained:
            if args.set == 'cifar100':
                ckpt = torch.load('/public/ly/xianyu/pretrained_model/resnet56_KD_15_c100/cifar100/K5_resnet56_KD_15_c100_cifar100_70.03.pt', map_location='cuda:%d' % args.gpu)
                model.load_state_dict(ckpt)
    elif args.arch == 'resnet56_KD_19_c100':
        model = resnet56_KD_19_c100(num_classes=100)
        if args.finetune:
            orginal_model = resnet56(num_classes=100)
            ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet56/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
            orginal_model.load_state_dict(ckpt)
            orginal_state_list = get_cresnet_layer_params(orginal_model)
            remain_list = [[0, 1, 2, 3, 4, 5], [2, 3, 4, 5], [1, 2, 3, 4, 5, 6, 7, 8]]
            model = load_cresnet_layer_params(orginal_state_list, model, remain_list, num_of_block=9)
            print('Load pretrained weights from the original model')
    elif args.arch == 'resnet56_KD_14':
        model = resnet56_KD_14(num_classes=args.num_classes)
        if args.finetune:
            orginal_model = resnet56(num_classes=args.num_classes)
            save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet56.th', map_location='cuda:%d' % args.gpu)
            ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
            orginal_model.load_state_dict(ckpt)
            orginal_state_list = get_cresnet_layer_params(orginal_model)
            remain_list = [[0, 1, 2, 4, 8], [0, 1, 2, 3], [5, 6, 7, 8]]
            model = load_cresnet_layer_params(orginal_state_list, model, remain_list, num_of_block=9)
            print('Load pretrained weights from the original model')
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/xianyu/pretrained_model/resnet56_KD_14/cifar10/K5_resnet56_KD_14_cifar10_93.83.pt', map_location='cuda:%d' % args.gpu)
                model.load_state_dict(ckpt)
    elif args.arch == 'resnet56_KD_13':
        model = resnet56_KD_13(num_classes=args.num_classes)
        if args.finetune:
            orginal_model = resnet56(num_classes=args.num_classes)
            save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet56.th', map_location='cuda:%d' % args.gpu)
            ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
            orginal_model.load_state_dict(ckpt)
            orginal_state_list = get_cresnet_layer_params(orginal_model)
            remain_list = [[0, 1, 2, 4, 7], [0, 1, 2], [5, 6, 7, 8]]
            model = load_cresnet_layer_params(orginal_state_list, model, remain_list, num_of_block=9)
            print('Load pretrained weights from the original model')
    elif args.arch == 'resnet56_KD_12':
        model = resnet56_KD_12(num_classes=args.num_classes)
        if args.finetune:
            orginal_model = resnet56(num_classes=args.num_classes)
            save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet56.th', map_location='cuda:%d' % args.gpu)
            ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
            orginal_model.load_state_dict(ckpt)
            orginal_state_list = get_cresnet_layer_params(orginal_model)
            remain_list = [[0, 1, 3, 5], [0, 1, 2], [5, 6, 7, 8]]
            model = load_cresnet_layer_params(orginal_state_list, model, remain_list, num_of_block=9)
            print('Load pretrained weights from the original model')
    elif args.arch == 'resnet56_KD_15':
        model = resnet56_KD_15(num_classes=args.num_classes)
        if args.finetune:
            orginal_model = resnet56(num_classes=args.num_classes)
            save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet56.th', map_location='cuda:%d' % args.gpu)
            ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
            orginal_model.load_state_dict(ckpt)
            orginal_state_list = get_cresnet_layer_params(orginal_model)
            remain_list = [[0, 1, 2, 3, 6, 7], [0, 1, 2, 3], [5, 6, 7, 8]]
            model = load_cresnet_layer_params(orginal_state_list, model, remain_list, num_of_block=9)
            print('Load pretrained weights from the original model')
    elif args.arch == 'resnet56_KD_16':
        model = resnet56_KD_16(num_classes=args.num_classes)
        if args.finetune:
            orginal_model = resnet56(num_classes=args.num_classes)
            save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet56.th', map_location='cuda:%d' % args.gpu)
            ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
            orginal_model.load_state_dict(ckpt)
            orginal_state_list = get_cresnet_layer_params(orginal_model)
            remain_list = [[0, 1, 2, 3, 6, 7], [0, 1, 2, 3, 4], [5, 6, 7, 8]]
            model = load_cresnet_layer_params(orginal_state_list, model, remain_list, num_of_block=9)
            print('Load pretrained weights from the original model')
    elif args.arch == 'ResNet50_13':
        model = ResNet50_13(pretrained=False)
        if args.finetune:
            orginal_model = resnet50(pretrained=True)
            orginal_state_list = get_Iresnet_layer_params(orginal_model)
            remain_list = [[0, 1, 2], [0, 1, 2, 3], [0, 1, 2], [1, 2]]
            model = load_Iresnet_layer_params(orginal_state_list, model, remain_list, num_of_block=[3, 4, 6, 3])
            print('Load pretrained weights from the original model')
    elif args.arch == 'cvgg8_bn':
        model = cvgg8_bn(num_classes=args.num_classes, batch_norm=True)
        if args.finetune:
            orginal_model = cvgg16_bn(num_classes=args.num_classes, batch_norm=True)
            ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg16_bn/cifar10/scores.pt', map_location='cuda:%d' % args.gpu)
            orginal_model.load_state_dict(ckpt)
            orginal_conv_list = [[0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40], [1, 4, 6]]
            pruned_conv_list = [
                [(0, 0), (3, 1), (7, 2), (11, 4), (15, -1), (19, 10), (22, 11), (25, 12)],
                [(1, 0), (4, 1), (6, 2)]]  # match the orginal_conv_list and pruned model
            model = load_vgg_layer_params(orginal_model, orginal_conv_list, model, pruned_conv_list)
            print('Load pretrained weights from the original model')
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/xianyu/pretrained_model/cvgg8_bn/cifar10/K5_cvgg8_bn_cifar10_93.44.pt', map_location='cuda:%d' % args.gpu)
                model.load_state_dict(ckpt)
    elif args.arch == 'cvgg8_bn_v1':
        model = cvgg8_bn_v1(num_classes=args.num_classes, batch_norm=True)
        if args.finetune:
            orginal_model = cvgg16_bn(num_classes=args.num_classes, batch_norm=True)
            ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg16_bn/cifar10/scores.pt', map_location='cuda:%d' % args.gpu)
            orginal_model.load_state_dict(ckpt)
            orginal_conv_list = [[0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40], [1, 4, 6]]
            pruned_conv_list = [
                [(0, 0), (3, 1), (7, 2), (10, 3), (14, 4), (18, 7), (21, 9), (25, 12)],
                [(1, 0), (4, 1), (6, 2)]]  # match the orginal_conv_list and pruned model
            model = load_vgg_layer_params(orginal_model, orginal_conv_list, model, pruned_conv_list)
            print('Load pretrained weights from the original model')
    else:
        assert "the model has not prepared"

    return model


