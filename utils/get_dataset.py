from data.Data import MNIST, CIFAR10, CIFAR100
from data.ImageNet_dali import ImageNetDali


def get_dataset(args):
    print(f"=> Getting {args.set} dataset")
    if args.set == 'imagenet_dali':
        dataset = ImageNetDali()
    elif args.set == 'mnist':
        dataset = MNIST()
    elif args.set == 'cifar10':
        dataset = CIFAR10()  # for normal training
    elif args.set == 'cifar100':
        dataset = CIFAR100()  # for normal training
    return dataset