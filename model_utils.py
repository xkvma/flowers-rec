import torch
from torch import nn
from torchvision.models import regnet_x_800mf

def modify_clsf_head(model, n_classes=5):
    out_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(out_features, 512),
        nn.Linear(512, n_classes)
    )
    return model

def freeze_backbone(model):
    for param in model.parameters() :
        param.requires_grad = False
    for param in model.fc.parameters() :
        param.requires_grad = True
    return model

def make_model(backbone_path):
    weights = torch.load(backbone_path, map_location="cpu", weights_only=True)
    model = regnet_x_800mf(weights=None)
    modify_clsf_head(model, 5)
    model.fc[-1] = nn.Identity()
    model.load_state_dict(weights)
    model.eval()
    return model