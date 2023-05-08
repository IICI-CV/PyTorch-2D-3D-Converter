import torch
import torch.nn as nn


def fix_2d_to_3d(size):
    if isinstance(size, tuple):
        size = [size[0], size[1], size[1]]
    return size


def convert2d_to_3d(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            conv3d = nn.Conv3d(
                module.in_channels,
                module.out_channels,
                kernel_size=fix_2d_to_3d(module.kernel_size),
                stride=fix_2d_to_3d(module.stride),
                padding=fix_2d_to_3d(module.padding),
            )
            setattr(model, name, conv3d)
        elif isinstance(module, nn.BatchNorm2d):
            # replace with a batchnorm3d layer
            batchnorm3d = nn.BatchNorm3d(fix_2d_to_3d(module.num_features))
            setattr(model, name, batchnorm3d)
        elif isinstance(module, nn.MaxPool2d):
            # replace with a maxpool3d layer
            maxpool3d = nn.MaxPool3d(
                kernel_size=fix_2d_to_3d(module.kernel_size),
                stride=fix_2d_to_3d(module.stride),
                padding=fix_2d_to_3d(module.padding),
            )
            setattr(model, name, maxpool3d)
        elif isinstance(module, nn.AvgPool2d):
            # replace with a avgpool3d layer
            avgpool3d = nn.AvgPool3d(
                kernel_size=fix_2d_to_3d(module.kernel_size),
                stride=fix_2d_to_3d(module.stride),
                padding=fix_2d_to_3d(module.padding),
            )
            setattr(model, name, avgpool3d)
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            # replace with a adaptiveavgpool3d layer
            adaptiveavgpool3d = nn.AdaptiveAvgPool3d(fix_2d_to_3d(module.output_size))
            setattr(model, name, adaptiveavgpool3d)
        else:
            # recurse for submodules
            convert2d_to_3d(module)
    return model


if __name__ == "__main__":
    import torchvision.models as models

    model = models.resnet18()
    model = convert2d_to_3d(model)
    a = torch.randn(1, 3, 16, 224, 224)
    print(model(a).shape)
