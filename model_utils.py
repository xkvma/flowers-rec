import torch
from torch import nn

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
    
