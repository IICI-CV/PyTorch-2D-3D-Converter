# PyTorch-2D-3D-Converter
Convert the 2D model in PyTorch into a 3D implementation.

⚠️Warning: This repository is not complete and cannot adapt to all conversion scenarios, and can only convert convolutional neural network models.

---

If there are fragments in the code similar to the following:
```python
class Net(nn.Module):
    """Temp Net"""
    def __init__(...):
        super(..., self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            mid_chs,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=groups,
            bias=bias,
            **kwargs,
        )
        ...
        
    def forward(self, x):
        B, C, H, W = x.shape # <---- like this !
        out = some_operations(x)
        ...
        ...
        return out
```
So you can use the following methods for conversion implementation:
```python
class Net(nn.Module):
    """Temp Net"""
    def __init__(...):
        super(..., self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            mid_chs,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=groups,
            bias=bias,
            **kwargs,
        )
        ...
        
    def forward(self, x):
        if isinstance(self.conv, nn.Conv2d):
          # 2D Conv
          B, C, H, W = x.shape # <---- like this !
          out = some_operations(x)
          ...
          ...
          return out
        else:
          # 3D Conv
          B, RC, D, H, W = x.shape # <---- like this !
          out = some_operations(x)
          ...
          # Replace the 2D operation with 3D, 
          # and note that the separate operations for H and W should be implemented simultaneously for D, H, and W.
          ...
          return out
```
---
