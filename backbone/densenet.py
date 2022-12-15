from torch import nn
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, resnet_fpn_backbone, BackboneWithFPN

class DenseNet(nn.Module):
    def __init__(self, num_classes):     # @Thuan: Add argumentation in __init__
        super(DenseNet, self).__init__()
        # @Thuan: Initiation model here

        # End of model initiation

    def forward(self, x):
        # @Thuan: Feedforward step here

        return x    # End of feedforward function

