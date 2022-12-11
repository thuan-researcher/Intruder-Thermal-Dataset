import argparse
import torch
from detector import *
from backbone import *
from loss import *
from data import Therin
import utils
from datetime import datetime
import os
from detector.fasterRCNN import FasterRCNN
from backbone.densenet import DenseNet
from engine import train_one_epoch, evaluate

parser = argparse.ArgumentParser("Intruder_Thermal_Dataset")

# Model Settings
parser.add_argument('--detector', type=str, default='fasterRCNN', help='detector name')
parser.add_argument('--backbone', type=str, default='densenet', help='backbone name')
parser.add_argument('--loss', type=str, default='focalloss', help='loss name')
parser.add_argument('--modelscale', type=float, default=1.0, help='model scale')

# Training Settings
parser.add_argument('--batch', type=int, default=64, help='batch size')
parser.add_argument('--epoch', type=int, default=50, help='epochs number')  
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
    os.makedirs(timestr)

    num_classes = 5
    # Load data
    train_dataset = Therin(args.data_dir, 'train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=args.batch,
                                                    shuffle=True,
                                                    num_workers=4)
    test_dataset = Therin(args.data_dir, 'test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=args.batch,
                                                    shuffle=True,
                                                    num_workers=4)

    # Create model
    backbone = DenseNet()
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
        train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=1)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, test_dataloader, device=device)


if __name__ == '__main__':
    main()