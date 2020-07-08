# MobileNeXt
A TensorFlow and PyTorch implementation of MobileNeXt

# Overview
A TensorFlow and PyTorch implementation of MobileNeXt architecture: [Rethinking Bottleneck Structure for Efficient Mobile Network Design](https://arxiv.org/pdf/2007.02269.pdf).
The authors rethink the necessity of inverted residual block and find it may bring risks of information loss and gradient confusion. They thus propose to flip the structure and present a novel bottleneck design, called the sandglass block, that performs identity mapping and spatial transformation at higher dimensions and thus alleviates information loss and gradient confusion effectively. In ImageNet classification, by simply replacing the inverted residual block with the sandglass block without increasing parameters and computation, the classification accuracy can be improved by more than 1.7% over MobileNetV2.

# Disclaimer
This is not the official implementation of MobileNeXt.
