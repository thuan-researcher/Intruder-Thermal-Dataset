# Intruder-Thermal-Dataset
Evaluation experiments on thermal imaging dataset for intruder detection task. 

## Installation
### Clone this repo or download source code
```
git clone https://github.com/thuan-researcher/Intruder-Thermal-Dataset.git
```
### Install requirement packages
``` 
pip install requirements.txt
```
## Data description
The dataset folder `./dataset` consists of two sub sets `Sync_train_img` and `Sync_test_img`:
- `Sync_train_img`: 10,000 images under `.BMP` format, each image contain only one object.
- `Sync_test_img`: 2,500 images under `.BMP` format, each image contain only one object.
Labels for those images are stored in files `sync_train_anno.json` and `sync_test_anno.json`. Each contains a list of annotation directory following format as: 
```{"image_id": i, "bbox", [x, y, w, h], "class": c}```
where i is the image index (file name), bbox is the bounding box, class (0-4) is the type of position (0: creeping, 1: crawling, 2: stooping, 3: climbing, 4: other).

![img01](./img/00001.JPG)![img02](./img/00002.JPG)![img03](./img/00003.JPG)![img04](./img/00004.JPG)
![img05](./img/00005.JPG)![img06](./img/00006.JPG)![img07](./img/00007.JPG)![img08](./img/00008.JPG)
![img09](./img/00009.JPG)![img10](./img/00010.JPG)![img11](./img/00011.JPG)![img12](./img/00012.JPG)

## Training and test
``` 
python3 train.py
```