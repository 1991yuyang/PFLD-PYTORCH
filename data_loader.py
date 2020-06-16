from torch.utils import data
from torchvision import transforms as T
from PIL import Image
import torch as t


class MySet(data.Dataset):

    def __init__(self, is_train):
        self.transformer = T.ToTensor()
        if is_train:
            with open("data/train_data/list.txt", "r", encoding="utf-8") as file:
                self.lines = file.read().strip().split("\n")
        else:
            with open("data/test_data/list.txt", "r", encoding="utf-8") as file:
                self.lines = file.read().strip().split("\n")

    def __getitem__(self, index):
        line = self.lines[index].split(" ")
        img_path = line[0]
        landmark = list(map(float, line[1:197]))  # 归一化后的关键点坐标
        attribute = list(map(float, line[197:203]))  # 6个人脸类型的01二值编码
        euler_angle = list(map(float, line[203:]))  # 3个欧拉角
        img = self.transformer(Image.open(img_path))
        return img, t.tensor(landmark).view((98, -1)).type(t.FloatTensor), t.tensor(attribute).type(t.FloatTensor), t.tensor(euler_angle).type(t.FloatTensor)

    def __len__(self):
        return len(self.lines)


if __name__ == "__main__":
    s = MySet(True)
    img, landmark, attribute, euler_angle = s[0]
    print(img.size())
    print(landmark)
    print(attribute)
    print(euler_angle)