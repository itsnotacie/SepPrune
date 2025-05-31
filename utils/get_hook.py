# from model.VGG_cifar import *
# from args import args

cfgs = {
    'ResNet18': [2, 2, 2, 2],
    'ResNet34': [3, 4, 6, 3],
    'ResNet50': [3, 4, 6, 3],
    'ResNet101': [3, 4, 23, 3],
    'ResNet152': [3, 8, 36, 3],
    'resnet20': [3, 3, 3],
    'resnet32': [5, 5, 5],
    'resnet44': [7, 7, 7],
    'resnet56': [9, 9, 9],
    'resnet110': [18, 18, 18],
    'Ivgg11_bn': ['features.0', 'features.4', 'features.8', 'features.11', 'features.15', 'features.18', 'features.22', 'features.25'],
    'Ivgg13_bn': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.21', 'features.24', 'features.28', 'features.31'],
    'Ivgg16_bn': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20', 'features.24', 'features.27', 'features.30', 'features.34', 'features.37', 'features.40'],
    'Ivgg19_bn': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17','features.20', 'features.23', 'features.27', 'features.30', 'features.33', 'features.36', 'features.40', 'features.43', 'features.46', 'features.49'],
    'cvgg11_bn': ['features.0', 'features.4', 'features.8', 'features.11', 'features.15', 'features.18', 'features.22',
               'features.25'],
    'cvgg13_bn': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.21',
               'features.24', 'features.28', 'features.31'],
    'cvgg16_bn': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20',
               'features.24', 'features.27', 'features.30', 'features.34', 'features.37', 'features.40'],
    'cvgg19_bn': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20',
               'features.23', 'features.27', 'features.30', 'features.33', 'features.36', 'features.40', 'features.43',
               'features.46', 'features.49'],
    'swin_base_patch4_window7_224': ['stages.0.blocks.0', 'stages.0.blocks.1', 'stages.0.downsample', 'stages.1.blocks.0',
                                     'stages.1.blocks.1', 'stages.1.downsample', 'stages.2.blocks.0', 'stages.2.blocks.1',
                                     'stages.2.blocks.2', 'stages.2.blocks.3', 'stages.2.blocks.4',
                                     'stages.2.blocks.5', 'stages.2.blocks.6', 'stages.2.blocks.7',
                                     'stages.2.blocks.8', 'stages.2.blocks.9',  'stages.2.blocks.10',
                                     'stages.2.blocks.11', 'stages.2.blocks.12', 'stages.2.blocks.13',
                                     'stages.2.blocks.14', 'stages.2.blocks.15', 'stages.2.blocks.16',
                                     'stages.2.blocks.17', 'stages.2.downsample', 'stages.3.blocks.0', 'stages.3.blocks.1'],
    'swin_large_patch4_window7_224': ['stages.0.blocks.0', 'stages.0.blocks.1', 'stages.0.downsample', 'stages.1.blocks.0',
                                      'stages.1.blocks.1', 'stages.1.downsample', 'stages.2.blocks.0', 'stages.2.blocks.1',
                                      'stages.2.blocks.2', 'stages.2.blocks.3', 'stages.2.blocks.4',
                                      'stages.2.blocks.5', 'stages.2.blocks.6', 'stages.2.blocks.7',
                                      'stages.2.blocks.8', 'stages.2.blocks.9', 'stages.2.blocks.10',
                                      'stages.2.blocks.11', 'stages.2.blocks.12', 'stages.2.blocks.13',
                                      'stages.2.blocks.14', 'stages.2.blocks.15', 'stages.2.blocks.16',
                                      'stages.2.blocks.17', 'stages.2.downsample', 'stages.3.blocks.0', 'stages.3.blocks.1'],
    'swin_small_patch4_window7_224': ['stages.0.blocks.0', 'stages.0.blocks.1', 'stages.0.downsample', 'stages.1.blocks.0',
                                      'stages.1.blocks.1', 'stages.1.downsample', 'stages.2.blocks.0', 'stages.2.blocks.1',
                                      'stages.2.blocks.2', 'stages.2.blocks.3', 'stages.2.blocks.4',
                                      'stages.2.blocks.5', 'stages.2.blocks.6', 'stages.2.blocks.7',
                                      'stages.2.blocks.8', 'stages.2.blocks.9',  'stages.2.blocks.10',
                                      'stages.2.blocks.11', 'stages.2.blocks.12', 'stages.2.blocks.13',
                                      'stages.2.blocks.14', 'stages.2.blocks.15', 'stages.2.blocks.16',
                                      'stages.2.blocks.17', 'stages.2.downsample', 'stages.3.blocks.0', 'stages.3.blocks.1'],
    'swin_tiny_patch4_window7_224': ['stages.0.blocks.0', 'stages.0.blocks.1', 'stages.0.downsample',
                                     'stages.1.blocks.0', 'stages.1.blocks.1', 'stages.1.downsample',
                                     'stages.2.blocks.0', 'stages.2.blocks.1', 'stages.2.blocks.2',
                                     'stages.2.blocks.3', 'stages.2.blocks.4', 'stages.2.blocks.5',
                                     'stages.2.downsample', 'stages.3.blocks.0', 'stages.3.blocks.1'],
    'vit_large_patch16_224': [f'blocks.{i}' for i in range(24)],
    'vit_base_patch16_224': [f'blocks.{i}' for i in range(12)],
    'vit_base_patch16_224mae': [f'blocks.{i}' for i in range(12)],
    'vit_base_patch16_224mocov3': [f'blocks.{i}' for i in range(12)],
    'vit_small_patch16_224': [f'blocks.{i}' for i in range(12)],
    'vit_small_patch16_224mocov3': [f'blocks.{i}' for i in range(12)],
    'vit_tiny_patch16_224': [f'blocks.{i}' for i in range(12)],
    'tdanet':[f'sm.unet.loc_glo_fus.{i}'for i in range(5)]
    
}

def get_inner_feature_for_resnet(model, hook, arch):
    handle_list = []
    cfg = cfgs[arch]
    if args.multigpu is not None:
        for i in range(len(cfg)):
            cfg[i] = 'module.' + cfg[i]
    print('cfg:', cfg)
    handle = model.conv1.register_forward_hook(hook)  # here!!!
    handle_list.append(handle)
    # handle.remove()  # free memory
    for i in range(cfg[0]):
        handle = model.layer1[i].register_forward_hook(hook)
        handle_list.append(handle)
    for i in range(cfg[1]):
        handle = model.layer2[i].register_forward_hook(hook)
        handle_list.append(handle)
    for i in range(cfg[2]):
        handle = model.layer3[i].register_forward_hook(hook)
        handle_list.append(handle)
    for i in range(cfg[3]):
        handle = model.layer4[i].register_forward_hook(hook)
        handle_list.append(handle)
    return handle_list


def get_inner_feature_for_smallresnet(model, hook, arch):
    handle_list = []
    cfg = cfgs[arch]
    if args.multigpu is not None:
        for i in range(len(cfg)):
            cfg[i] = 'module.' + cfg[i]
    print('cfg:', cfg)
    handle = model.conv1.register_forward_hook(hook)
    handle_list.append(handle)
    # handle.remove()  # free memory
    for i in range(cfg[0]):
        handle = model.layer1[i].register_forward_hook(hook)
        handle_list.append(handle)
    for i in range(cfg[1]):
        handle = model.layer2[i].register_forward_hook(hook)
        handle_list.append(handle)
    for i in range(cfg[2]):
        handle = model.layer3[i].register_forward_hook(hook)
        handle_list.append(handle)
    return handle_list


def get_inner_feature_for_vgg(model, hook, arch):
    cfg = cfgs[arch]
    if args.multigpu is not None:
        for i in range(len(cfg)):
            cfg[i] = 'module.' + cfg[i]

    handle_list = []
    print('cfg:', cfg)
    count = 0
    for idx, m in enumerate(model.named_modules()):
        name, module = m[0], m[1]
        if count < len(cfg):
            if name == cfg[count]:
                print(module)
                handle = module.register_forward_hook(hook)
                handle_list.append(handle)
                count += 1
        else:
            break
    return handle_list


def get_inner_feature_for_tdanet(model, hook, arch):
    cfg = cfgs[arch]
    # if args.multigpu is not None:
    #     for i in range(len(cfg)):
    #         cfg[i] = 'module.' + cfg[i]

    handle_list = []
    print('cfg:', cfg)
    count = 0
    for idx, m in enumerate(model.named_modules()):
        name, module = m[0], m[1]
        # print(name)
        # print(type(cfg))
        if count < len(cfg):
            if name==cfg[count]:
                print(name,cfg[count])
                handle = module.register_forward_hook(hook)
                handle_list.append(handle)
                count += 1
        else:
            break
    # print(len(handle_list))
    # print(handle_list)
    return handle_list

if __name__ == "__main__":
    # demo
    import torch
    from torchvision.models import *

    input = torch.randn((2, 3, 224, 224))
    inter_feature = []
    model = vgg11_bn()

    def hook(module, input, output):
        inter_feature.append(output.clone().detach())
    get_inner_feature_for_vgg(model, hook, 'cvgg19')
    model(input)
