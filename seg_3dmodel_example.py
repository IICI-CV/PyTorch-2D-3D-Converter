import torch
from torch import nn
import segmentation_models_pytorch as smp
from converter import convert2d_to_3d

if __name__ == "__main__":
    model = smp.Unet(
        encoder_name="resnet101",
        encoder_weights=None,
        in_channels=3,
        classes=4,
    )
    a = torch.randn(1, 3, 224, 224)
    print(model(a).shape) # torch.Size([1, 4, 224, 224])
    
    # Convert to 3D Model
    
    model_3d = convert2d_to_3d(model)
    a = torch.randn(1, 3, 64, 64, 64)
    print(model_3d(a).shape) # torch.Size([1, 4, 64, 64, 64])
    
