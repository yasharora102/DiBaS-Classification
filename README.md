# DiBaS-Classification

Download dataset & Pretrained (resnet18) weights [here.](https://drive.google.com/drive/folders/1FYsXQiVGfoi72hmut6wA3kUxQKDZmJWm) 

### Or

```bash
pip install gdown
gdown https://drive.google.com/drive/folders/1FYsXQiVGfoi72hmut6wA3kUxQKDZmJWm --folder

```
### Unzip everything and place in corresponding folders

### Your Final file structure should look like this.
```
├── Checkpoints
│   └── model_best.pth.tar
├── dataset.py
├── main.py
├── splitted
│   ├── train
│   └── val
└── train_test.split.py
```
### Run the testing script
```python
python train.py
```

