import torch as t
from torch import nn
import os
from PIL import Image
from torchvision import transforms as T
from model import BackBone
import cv2
import numpy as np
import shutil
from camera import camera

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
with open("./config.json", "r", encoding="utf-8") as file:
    conf = eval(file.read())
predict_conf = conf["predict"]
is_video = predict_conf["is_video"]
video_path = predict_conf["video_path"]
train_conf = conf["train"]
use_best_model = predict_conf["use_best_model"]
use_camera = predict_conf["use_camera"]
model_save_path = train_conf["model_save_path"]
test_img_dir = predict_conf["test_img_dir"]
predict_result_save_dir = predict_conf["predict_result_save_dir"]
if os.path.exists(predict_result_save_dir):
    if os.listdir(predict_result_save_dir):
        shutil.rmtree(predict_result_save_dir)
        os.mkdir(predict_result_save_dir)



def load_model(use_best_model):
    model = BackBone()
    model = nn.DataParallel(module=model, device_ids=[0])
    model = model.cuda(0)
    if use_best_model:
        print("load best model......")
        model.load_state_dict(t.load(os.path.join(model_save_path, "best_model.pth")))
    else:
        print("load epoch model......")
        model.load_state_dict(t.load(os.path.join(model_save_path, "epoch_model.pth")))
    model.eval()
    return model


def predict_one_img(img_path, model):
    """

    :param img_path: 人脸图像路径
    :param model: 模型
    :return:
    """
    cv2_img = cv2.imread(img_path)
    img_shape = cv2_img.shape[:2]
    img = cv2.resize(cv2_img, (112, 112))
    img = Image.fromarray(img)
    transformer = T.ToTensor()
    img = transformer(img).unsqueeze(0)
    landmark, _ = model(img)
    landmark = landmark.cpu().detach().numpy().reshape(-1, 2)
    landmark = landmark * np.array(img_shape[::-1])
    return landmark


if __name__ == "__main__":
    model = load_model(use_best_model)
    if not use_camera:
        img_names = os.listdir(test_img_dir)
        for img_name in img_names:
            result_save_path = os.path.join(predict_result_save_dir, img_name)
            img_path = os.path.join(test_img_dir, img_name)
            landmark = predict_one_img(img_path, model)
            img = cv2.imread(img_path)
            for point in landmark:
                cv2.circle(img, tuple(point.astype(np.int)), 0, (0, 0, 255), 2)
            cv2.imwrite(result_save_path, img)
    else:
        camera(model, video_path, is_video)