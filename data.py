import torch
import numpy as np
import transforms as T
from torch.utils.data import Dataset
from PIL import Image
from torchvision.ops.boxes import box_convert
import glob
import json

# dataset
#       |_ train
#       |   |_ .BMP files
#       |   |_ annotation
#       |           |_ classes.txt (one class per line)
#       |           |_ .txt anno files (class x_center y_center width height)
#       |_ test
#       |_ val

class Therin(Dataset):   # Therin: Intruder thermal dataset
    def __init__(self, dir, set):
        
        self._dir = dir + '/Sync_' + set + '_img'
        self._imglist = glob.glob(self._dir + '/*.BMP')
        self._json_path = dir + '/Sync_' + set + '_anno.json'
        with open(self._json_path) as anno_file:
            self._anno = json.load(anno_file)
        self._transform = T.Compose([T.ToTensor()])

    def __len__(self):
        return len(self._imglist)

    def __getitem__(self, index):
        image = Image.open(self._imglist[index])

        boxes = np.zeros((1, 4), dtype=np.float32)
        boxes[0] = self._anno[index]['bbox']
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        boxes = box_convert(boxes, in_fmt='xywh', out_fmt='xyxy')

        gt_classes = np.zeros((1), dtype=np.int32)
        gt_classes[0] = self._anno[index]['class']
        gt_classes = torch.as_tensor(gt_classes, dtype=torch.int64)

        image_id = self._anno[index]['image_id']
        image_id = torch.as_tensor(image_id, dtype=torch.int64)

        target = {"labels": gt_classes, "boxes": boxes, "image_id": image_id, "area": torch.Tensor(0), "iscrowd": torch.Tensor(0)}
        image, target = self._transform(image, target)
        return image, target

    def collate_fn(self, batch):
        return tuple(zip(*batch))
