import argparse
import os
import torch
from detector import *
from backbone import *
from loss import *
from data import Therin
from datetime import datetime
from detector.fasterRCNN import FasterRCNN
from backbone.densenet import DenseNet
from .utils import *
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _resnet_fpn_extractor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_V2_Weights

parser = argparse.ArgumentParser("Intruder_Thermal_Dataset")

# Model Settings
parser.add_argument('--detector', type=str, default='fasterRCNN', help='detector name')
parser.add_argument('--backbone', type=str, default='densenet', help='backbone name')
parser.add_argument('--loss', type=str, default='focalloss', help='loss name')
parser.add_argument('--modelscale', type=float, default=1.0, help='model scale')

# Training Settings
parser.add_argument('--batch', type=int, default=4, help='batch size')
parser.add_argument('--epoch', type=int, default=10, help='epochs number')  
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--decay', type=float, default=3e-4, help='weight decay')

# Dataset Settings
parser.add_argument('--data_dir', type=str, default='./dataset', help='dataset dir')


args = parser.parse_args()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    timestr = datetime.now().strftime("%Y%m%d-%H%M%S%f")
    model_save_dir = timestr
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    num_classes = 5
    # Load data
    train_dataset = Therin(args.data_dir, 'train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=args.batch,
                                                    shuffle=True,
                                                    num_workers=4,
                                                    collate_fn=train_dataset.collate_fn)
    test_dataset = Therin(args.data_dir, 'test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=args.batch,
                                                    shuffle=True,
                                                    num_workers=4,
                                                    collate_fn=test_dataset.collate_fn)

    # Create model
    backbone = resnet_fpn_backbone('resnet18', False)
    model = FasterRCNN(backbone, num_classes)
    model.to(device)

    # Define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr,
                                momentum=args.momentum, weight_decay=args.decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    
    #Training
    for epoch in range(args.epoch):
        # train for one epoch
        loss_dict, total_loss = train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=1)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        _, mAP = evaluate(model, test_dataloader, device=device)
        print('validation mAp is {}'.format(mAP))
        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'loss_dict': loss_dict,
            'total_loss': total_loss}
        torch.save(save_files,
                    os.path.join(model_save_dir, "{}-model-{}-mAp-{}.pth".format(args.backbone, epoch, mAP)))

if __name__ == '__main__':
    main()