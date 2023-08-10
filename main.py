import os
import random
from dataset import Dataset
from model import VisionTransformerModel
from train import train


if __name__ == '__main__':
    # file list
    file_list = []
    for root, dirs, files in os.walk('.', topdown=False):
        for name in files:
            if not name.endswith('.jpg'):
                continue

            file_list.append(os.path.join(root, name))

    # shuffle file list
    random.shuffle(file_list)

    # train and valid split
    train_size = int(0.7 * len(file_list))

    train_list = file_list[:train_size]
    valid_list = file_list[train_size:]

    train_dataset = Dataset(train_list)
    valid_dataset = Dataset(valid_list)

    # train
    ViT = VisionTransformerModel(img_size=(28, 28), patch_size=(4, 4), pos_emb_size=16, num_classes=10)

    train(model=ViT, epochs=100, train_dataset=train_dataset, valid_dataset=valid_dataset, learning_rate=0.01)
