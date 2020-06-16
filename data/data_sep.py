# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import shutil
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from utils import calculate_pitch_yaw_roll

debug = False


def rotate(angle, center, landmark):
    rad = angle * np.pi / 180.0
    alpha = np.cos(rad)
    beta = np.sin(rad)
    M = np.zeros((2, 3), dtype=np.float32)
    M[0, 0] = alpha
    M[0, 1] = beta
    M[0, 2] = (1 - alpha) * center[0] - beta * center[1]
    M[1, 0] = -beta
    M[1, 1] = alpha
    M[1, 2] = beta * center[0] + (1 - alpha) * center[1]

    landmark_ = np.asarray([(M[0, 0] * x + M[0, 1] * y + M[0, 2],
                             M[1, 0] * x + M[1, 1] * y + M[1, 2]) for (x, y) in landmark])
    return M, landmark_


class ImageDate():
    def __init__(self, line, imgDir, image_size=112):
        self.image_size = image_size
        line = line.strip().split()
        # 0-195: landmark 坐标点  196-199: bbox 坐标点;
        # 200: 姿态(pose)         0->正常姿态(normal pose)          1->大的姿态(large pose)
        # 201: 表情(expression)   0->正常表情(normal expression)    1->夸张的表情(exaggerate expression)
        # 202: 照度(illumination) 0->正常照明(normal illumination)  1->极端照明(extreme illumination)
        # 203: 化妆(make-up)      0->无化妆(no make-up)             1->化妆(make-up)
        # 204: 遮挡(occlusion)    0->无遮挡(no occlusion)           1->遮挡(occlusion)
        # 205: 模糊(blur)         0->清晰(clear)                    1->模糊(blur)
        # 206: 图片名称
        assert (len(line) == 207)
        self.list = line
        self.landmark = np.asarray(list(map(float, line[:196])), dtype=np.float32).reshape(-1, 2)
        self.box = np.asarray(list(map(int, line[196:200])), dtype=np.int32)
        flag = list(map(int, line[200:206]))
        flag = list(map(bool, flag))
        self.pose = flag[0]
        self.expression = flag[1]
        self.illumination = flag[2]
        self.make_up = flag[3]
        self.occlusion = flag[4]
        self.blur = flag[5]
        self.path = os.path.join(imgDir, line[206])
        self.img = None

        self.imgs = []
        self.landmarks = []
        self.boxes = []

    def load_data(self, is_train, repeat, mirror=None):
        if (mirror is not None):
            with open(mirror, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 1
                mirror_idx = lines[0].strip().split(',')
                mirror_idx = list(map(int, mirror_idx))
        xy = np.min(self.landmark, axis=0).astype(np.int32)  # 当前人脸中包含所有关键点的最小外接矩形的左上角点坐标
        zz = np.max(self.landmark, axis=0).astype(np.int32)  # 当前人脸中包含所有关键点的最小外接矩形的右下角点坐标
        wh = zz - xy + 1  # 当前人脸包含所有关键点的最小外接矩形的宽和高

        center = (xy + wh / 2).astype(np.int32)  # 当前人脸包含所有关键点的最小外接矩形的中心坐标点
        img = cv2.imread(self.path)
        boxsize = int(np.max(wh) * 1.2)  # 将当前人脸包含所有关键点的最小外接矩形宽高扩充为原来的1.2倍
        xy = center - boxsize // 2  # 当前人脸关键点最小外接矩形大小扩充1.2倍后的左上角坐标
        x1, y1 = xy  # 当前人脸关键点最小外接矩形大小扩充1.2倍后的左上角坐标，有可能为负值
        x2, y2 = xy + boxsize  # 当前人脸关键点最小外接矩形大小扩充1.2倍后右下角坐标，有可能超出原图的大小
        height, width, _ = img.shape  # 原图高宽
        # 下面就是判断当前人脸关键点最小外接矩形在扩充为1.2倍后左上角点和右下角点是否超出了原图的左上角或右下角，
        # 如果超出则令最小外接矩形的左上角点为原图左上角点或最小外接矩形的右下角点为图像右下角点
        dx = max(0, -x1)  # 计算扩充1.2倍后的最小外接矩形左上角点是否在图像原点左侧，并返回相对于图像原点靠左偏移量
        dy = max(0, -y1)  # 计算扩充1.2倍后的最小外接矩形左上角点是否在图像原点上方，并返回相对于图像原点向上偏移量
        x1 = max(0, x1)  # 如果扩充1.2倍的最小外接矩形左上角点在图像原点左侧，则令左上角点横坐标为0
        y1 = max(0, y1)  # 如果扩充1.2倍的最小外接矩形左上角点在图像原点上方，则令左上角点纵坐标为0

        edx = max(0, x2 - width)  # 计算扩充1.2倍后的最小外接矩形右下角点是否在图像右下角的右侧，并返回相对于图像右下角向右偏移量
        edy = max(0, y2 - height)  # 计算扩充1.2倍后的最小外接矩形右下角点是否在图像右下角的下方，并返回相对于图像右下角向下偏移量
        x2 = min(width, x2)  # 如果扩充1.2倍后最小外接矩形在图像右下角的右侧，则令最小外接矩形右下角的横坐标为原图宽度
        y2 = min(height, y2)  # 如果扩充1.2倍后最小外接矩形在图像右下角的下方，则令最小外接矩形右下角的纵坐标为原图高度
        imgT = img[y1:y2, x1:x2]  # 从原图中按照扩充后最小外接矩形将当前人脸截出
        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            # 将截出的人脸边缘扩充以弥补扩充1.2倍后左上角点或右下角点超出的部分
            imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
        if imgT.shape[0] == 0 or imgT.shape[1] == 0:
            print("=====================")
            imgTT = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for x, y in (self.landmark + 0.5).astype(np.int32):
                cv2.circle(imgTT, (x, y), 1, (0, 0, 255))
            cv2.imshow('0', imgTT)
            if cv2.waitKey(0) == 27:
                exit()
        imgT = cv2.resize(imgT, (self.image_size, self.image_size))  # 将边缘弥补后的当前截取的人脸resize成112大小
        landmark = (self.landmark - xy) / boxsize  # 关键点在当前截取的人脸图像（边缘扩充弥补后，resize之前）上的归一化后的坐标
        assert (landmark >= 0).all(), str(landmark) + str([dx, dy])
        assert (landmark <= 1).all(), str(landmark) + str([dx, dy])
        self.imgs.append(imgT)
        self.landmarks.append(landmark)

        if is_train:
            while len(self.imgs) < repeat:
                angle = np.random.randint(-30, 30)
                cx, cy = center
                cx = cx + int(np.random.randint(-boxsize * 0.1, boxsize * 0.1))
                cy = cy + int(np.random.randint(-boxsize * 0.1, boxsize * 0.1))
                M, landmark = rotate(angle, (cx, cy), self.landmark)

                imgT = cv2.warpAffine(img, M, (int(img.shape[1] * 1.1), int(img.shape[0] * 1.1)))

                wh = np.ptp(landmark, axis=0).astype(np.int32) + 1
                size = np.random.randint(int(np.min(wh)), np.ceil(np.max(wh) * 1.25))
                xy = np.asarray((cx - size // 2, cy - size // 2), dtype=np.int32)
                landmark = (landmark - xy) / size
                if (landmark < 0).any() or (landmark > 1).any():
                    continue

                x1, y1 = xy
                x2, y2 = xy + size
                height, width, _ = imgT.shape
                dx = max(0, -x1)
                dy = max(0, -y1)
                x1 = max(0, x1)
                y1 = max(0, y1)

                edx = max(0, x2 - width)
                edy = max(0, y2 - height)
                x2 = min(width, x2)
                y2 = min(height, y2)

                imgT = imgT[y1:y2, x1:x2]
                if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                    imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)

                imgT = cv2.resize(imgT, (self.image_size, self.image_size))

                if mirror is not None and np.random.choice((True, False)):
                    landmark[:, 0] = 1 - landmark[:, 0]
                    landmark = landmark[mirror_idx]
                    imgT = cv2.flip(imgT, 1)
                self.imgs.append(imgT)
                self.landmarks.append(landmark)

    def save_data(self, path, prefix):
        attributes = [self.pose, self.expression, self.illumination, self.make_up, self.occlusion, self.blur]
        attributes = np.asarray(attributes, dtype=np.int32)
        attributes_str = ' '.join(list(map(str, attributes)))
        labels = []
        TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
        for i, (img, lanmark) in enumerate(zip(self.imgs, self.landmarks)):
            assert lanmark.shape == (98, 2)
            save_path = os.path.join(path, prefix + '_' + str(i) + '.png')
            assert not os.path.exists(save_path), save_path
            cv2.imwrite(save_path, img)

            euler_angles_landmark = []
            for index in TRACKED_POINTS:
                euler_angles_landmark.append(lanmark[index])
            euler_angles_landmark = np.asarray(euler_angles_landmark).reshape((-1, 28))
            pitch, yaw, roll = calculate_pitch_yaw_roll(euler_angles_landmark[0])
            euler_angles = np.asarray((pitch, yaw, roll), dtype=np.float32)
            euler_angles_str = ' '.join(list(map(str, euler_angles)))

            landmark_str = ' '.join(list(map(str, lanmark.reshape(-1).tolist())))

            label = '{} {} {} {}\n'.format(save_path, landmark_str, attributes_str, euler_angles_str)

            labels.append(label)
        return labels


def get_dataset_list(imgDir, outDir, landmarkDir, is_train):
    with open(landmarkDir, 'r') as f:
        lines = f.readlines()
        labels = []
        save_img = os.path.join(outDir, 'imgs')
        if not os.path.exists(save_img):
            os.mkdir(save_img)

        if debug:
            lines = lines[:100]
        for i, line in enumerate(lines):
            Img = ImageDate(line, imgDir)
            img_name = Img.path
            Img.load_data(is_train, 10, Mirror_file)
            _, filename = os.path.split(img_name)
            filename, _ = os.path.splitext(filename)
            label_txt = Img.save_data(save_img, str(i) + '_' + filename)
            labels.append(label_txt)
            if ((i + 1) % 100) == 0:
                print('file: {}/{}'.format(i + 1, len(lines)))

    with open(os.path.join(outDir, 'list.txt'), 'w') as f:
        for label in labels:
            f.writelines(label)


if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    imageDirs = 'WFLW/WFLW_images'
    Mirror_file = 'WFLW/WFLW_annotations/Mirror98.txt'

    landmarkDirs = ['WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt',
                    'WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt']

    outDirs = ['test_data', 'train_data']
    for landmarkDir, outDir in zip(landmarkDirs, outDirs):
        outDir = os.path.join(root_dir, outDir)
        print(outDir)
        if os.path.exists(outDir):
            shutil.rmtree(outDir)
        os.mkdir(outDir)
        if 'list_98pt_rect_attr_test.txt' in landmarkDir:
            is_train = False
        else:
            is_train = True
        imgs = get_dataset_list(imageDirs, outDir, landmarkDir, is_train)
    print('end')