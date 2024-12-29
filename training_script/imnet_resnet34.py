import os

import argparse

parser = argparse.ArgumentParser('Training CIFAR')

parser.add_argument('--lr', default=0.01, type=float,
                        help='config name')

parser.add_argument('--gpu', default='0',type=str,
                        help='gpu')
parser.add_argument('--task', type=int,
                        help='task index')
parser.add_argument('-w', default=8,type=int,
                        help='model width')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import clip
import torch
from copy import deepcopy

import numpy as np

from utils import *
from models.resnets import resnet34im
import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import torch.backends.cudnn as cudnn
import pickle as pkl


def reset_random(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


reset_random(seed=args.task)

# INITIALIZATIONS

if __name__ == "__main__":

    task_split_dict = {k:list(range(k*200,(k+1)*200)) for k in range(5)}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_clip = True
    model_dir = f'./checkpoints/imnet_{"clip" if use_clip else "logits"}'

    models_per_run = 1  # num models to train per split
    # data_dir = './data/cifar-100-python'
    data_dir = "~/datasets/ILSVRC2012/"
    wrapper = torchvision.datasets.ImageNet
    num_classes = 1000  # num classes in dataset
    batch_size = 100  # batch size
    model_width = args.w                        # resnet model width
    epochs = 100  # train epochs

    model_dir = os.path.join(model_dir, f'resnet34imx{model_width}', 'pairsplits')
    print(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    img_size = 224
    train_transform= T.Compose([
            T.RandomResizedCrop(img_size,scale=(0.8,1)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_transform = T.Compose([
            T.Resize(int(img_size * 1.143)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_dset = wrapper(root=data_dir,
                         split="train",
                         transform=train_transform)
    test_dset = wrapper(root=data_dir,
                        split="val",
                        transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=8)

    if 'clip' in model_dir:
        clip_features = load_clip_features([v[0] for v in test_dset.classes], device=device)
        out_dim = 512
    else:
        out_dim = num_classes

    splits = task_split_dict[args.task]


    split_trainers = torch.utils.data.DataLoader(torch.utils.data.Subset(
            train_dset, [
                i for i, label in enumerate(train_dset.targets)
                if label in splits
            ]),
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=8)
    

    split_testers = torch.utils.data.DataLoader(torch.utils.data.Subset(
            test_dset, [
                i for i, label in enumerate(test_dset.targets)
                if label in splits
            ]),
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=8)
    
    label_remapping = np.zeros(num_classes, dtype=int)
    label_remapping[splits] = np.arange(len(splits))
    label_remapping = torch.from_numpy(label_remapping)
    print("label remapping: {}".format(label_remapping))
    print(f"{splits}")
    save_dir = os.path.join(model_dir, f"{args.task}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
            save_dir,
            f'resnet34imx{model_width}_v{len(os.listdir(save_dir))}.pth.tar')
    for j in range(models_per_run):
        model = resnet34im(num_classes=out_dim,w=model_width).cuda().train()
        # model = torchvision.models.resnet.resnet34(num_classes=out_dim).cuda().train()

        if 'clip' in model_dir:
            class_vectors = clip_features[splits]
            model, final_acc = train_cliphead(
                model=model,
                train_loader=split_trainers,
                test_loader=split_testers,
                class_vectors=class_vectors,
                remap_class_idxs=label_remapping,
                epochs=epochs,
                lr=args.lr,save_path=save_path,warm_epoch=1,sche_type="step")
        else:
            model, final_acc = train_logits(
                model=model,
                train_loader=split_trainers,
                test_loader=split_testers,
                epochs=epochs,
                lr=args.lr)

        print(f'Base model on {splits} Acc: {final_acc}')
        print('Saving Base Model')
        save_model(model, save_path)

    print('Done!')