import torchvision
from torchvision.models.detection import SSD
from torchvision.models.detection.rpn import AnchorGenerator

def ssd(backbone, class_num):
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    # put the pieces together inside a FasterRCNN model
    model = SSD(backbone,
                    num_classes=class_num,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler)
    return model