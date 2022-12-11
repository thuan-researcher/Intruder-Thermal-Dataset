from torch.utils.data import Dataset

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
        
        self._dir = dir
        self._set = set
    
    def _get_ann_file(self):
        return 0

    def __len__(self):
        return 0

    def __getitem__(self, index):
        classs = 0
        image = 0
        xmin = 0
        ymin = 0
        xmax = 0
        ymax = 0
        target = {"label": classs, "box": [xmin, ymin, xmax, ymax]}
        return image, target