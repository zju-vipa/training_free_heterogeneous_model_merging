from torch import nn
import timm
from .vit_clip import *
from .vision_transformer import VisionTransformer as VT
import torch

def vit_s_timm(num_classes=10, pretrained=True):

    net = timm.create_model("vit_small_patch16_384", pretrained=pretrained)
    net.head = nn.Linear(net.head.in_features, num_classes)
    return net

def vit_b_timm(num_classes=10, pretrained=True):
    net = timm.create_model("vit_base_patch16_224_clip_laion2b", pretrained=pretrained)
    if num_classes != net.head.out_features:
        net.head = nn.Linear(net.head.in_features, num_classes)
    else:
        print("use raw header", net.head.out_features)
    return net

def vit_b_dino(num_classes=10, pretrained=True):
    net = timm.create_model("vit_base_patch16_224_dino", pretrained=pretrained)
    net.head = nn.Linear(net.norm.normalized_shape[0], num_classes)
    nn.init.xavier_normal_(net.head.weight)
    nn.init.zeros_(net.head.bias)
    return net

def vit7(num_classes = 10,**kwargs):
    net = VT(img_size=32,patch_size=4,depth=7,mlp_ratio=1,num_heads=4,embed_dim=384,num_classes=num_classes)
    return net


def vit19(num_classes = 10,**kwargs):
    net = VT(img_size=32,patch_size=4,depth=19,mlp_ratio=1,num_heads=4,embed_dim=384,num_classes=num_classes)
    return net

def vit1h7(num_classes = 10,**kwargs):
    net = VT(img_size=32,patch_size=4,depth=7,mlp_ratio=1,num_heads=1,embed_dim=384,num_classes=num_classes)
    return net


def vit1h19(num_classes = 10,**kwargs):
    net = VT(img_size=32,patch_size=4,depth=19,mlp_ratio=1,num_heads=1,embed_dim=384,num_classes=num_classes)
    return net

def vithd(num_classes = 10,num_heads=1,depth=7,pretrained=True,**kwargs):
    net = VT(img_size=56,patch_size=8,depth=depth,mlp_ratio=1,num_heads=num_heads,embed_dim=384,num_classes=num_classes)
    # net0 = timm.create_model("vit_small_patch32_224.augreg_in21k_ft_in1k", pretrained=pretrained).state_dict()
    # net.load_state_dict({
    #     "cls_token": net0["cls_token"],
    #     "pos_embed": net0["pos_embed"]
    # }, strict=False)
    # net.pos_embed.requires_grad = False
    # net.cls_token.requires_grad = False

    net0 = torch.load("checkpoints_all/cifar100_clip_e250/vith4d7/pairsplits/0/vith4d7_v0.pth.tar", map_location="cpu")
    net.load_state_dict({
        "cls_token": net0["cls_token"],
        "pos_embed": net0["pos_embed"],
        "patch_embed.proj.weight": net0["patch_embed.proj.weight"],
        "patch_embed.proj.bias": net0["patch_embed.proj.bias"],
    }, strict=False)
    net.pos_embed.requires_grad = False
    net.cls_token.requires_grad = False
    net.patch_embed.proj.weight.requires_grad = False
    net.patch_embed.proj.bias.requires_grad = False
    
    return net


