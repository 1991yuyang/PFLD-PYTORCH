import torch as t
from torch import nn, optim
from model import BackBone, AuxNet
from data_loader import MySet
from loss import Loss
from torch.utils import data
import os
with open("./config.json", "r", encoding="utf-8") as file:
    train_conf = eval(file.read())["train"]
gpu_devices = train_conf["gpu_devices"]
device_ids = list(range(len(gpu_devices)))
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_devices))
model_save_path = train_conf["model_save_path"]
epoch = train_conf["epoch"]
batch_size = train_conf["batch_size"]
lr = train_conf["lr"]
lr_shrink_epoch = train_conf["lr_shrink_epoch"]
lr_shrink_rate = train_conf["lr_shrink_rate"]
weight_decay = train_conf["weight_decay"]
save_model_epoch = train_conf["save_model_epoch"]


def train():
    global lr
    current_minimum_valid_loss = float("inf")
    backbone_model = BackBone()
    aux_model = AuxNet()
    backbone_model = nn.DataParallel(module=backbone_model, device_ids=device_ids)
    aux_model = nn.DataParallel(module=aux_model, device_ids=device_ids)
    backbone_model.cuda(device_ids[0])
    aux_model.cuda(device_ids[0])
    optimizer = optim.Adam([
        {"params": backbone_model.parameters()},
        {"params": aux_model.parameters()}
    ], lr=lr, weight_decay=weight_decay)
    criterion = Loss().cuda(device_ids[0])
    for e in range(epoch):
        train_loader = iter(data.DataLoader(MySet(True), batch_size=batch_size, shuffle=True, drop_last=False))
        valid_loader = iter(data.DataLoader(MySet(False), batch_size=batch_size, shuffle=True, drop_last=False))
        step = 0
        all_step = len(train_loader)
        for img_train, landmark_train, attribute_train, euler_angle_train in train_loader:
            step += 1
            backbone_model.train()
            aux_model.train()
            img_train_cuda = img_train.cuda(device_ids[0])
            landmark_train_cuda = landmark_train.cuda(device_ids[0])
            attribute_train_cuda = attribute_train.cuda(device_ids[0])
            euler_angle_train_cuda = euler_angle_train.cuda(device_ids[0])
            landmark_pred, aux_input_feature = backbone_model(img_train_cuda)
            euler_angle_pred = aux_model(aux_input_feature)
            weight_train_loss, landmark_l2_distance = criterion(landmark_pred, euler_angle_pred, landmark_train_cuda, euler_angle_train_cuda, attribute_train_cuda, t.tensor(batch_size).type(t.FloatTensor).cuda(device_ids[0]))
            optimizer.zero_grad()
            weight_train_loss.backward()
            optimizer.step()
            try:
                img_valid, landmark_valid, attribute_valid, euler_angle_valid = next(valid_loader)
            except:
                valid_loader = iter(data.DataLoader(MySet(False), batch_size=batch_size, shuffle=True, drop_last=False))
                img_valid, landmark_valid, attribute_valid, euler_angle_valid = next(valid_loader)
            backbone_model.eval()
            aux_model.eval()
            img_valid_cuda = img_valid.cuda(device_ids[0])
            landmark_valid_cuda = landmark_valid.cuda(device_ids[0])
            attribute_valid_cuda = attribute_valid.cuda(device_ids[0])
            euler_angle_valid_cuda = euler_angle_valid.cuda(device_ids[0])
            with t.no_grad():
                landmark_pred_valid, aux_feature_input_valid = backbone_model(img_valid_cuda)
                euler_angle_pred_valid = aux_model(aux_feature_input_valid)
                weight_loss_valid, landmark_l2_distance_valid = criterion(landmark_pred_valid, euler_angle_pred_valid, landmark_valid_cuda, euler_angle_valid_cuda, attribute_valid_cuda, t.tensor(batch_size).type(t.FloatTensor).cuda(device_ids[0]))
            if weight_loss_valid.item() < current_minimum_valid_loss:
                current_minimum_valid_loss = weight_loss_valid.item()
                print("saving best model......")
                t.save(backbone_model.state_dict(), os.path.join(model_save_path, "best_model.pth"))
            print("epoch: %d, step: %d/%d, train_weight_loss: %.3f, valid_weight_loss: %.3f, train_l2_distance: %.3f, valid_l2_distance: %.3f" % (e, step, all_step, weight_train_loss.item(), weight_loss_valid.item(), landmark_l2_distance.item(), landmark_l2_distance_valid.item()))
        if e + 1 % lr_shrink_epoch == 0:
            lr *= 0.1
            optimizer = optim.Adam([
                {"params": backbone_model.parameters()},
                {"params": aux_model.parameters()}
            ], lr=lr, weight_decay=weight_decay)
        if e + 1 % save_model_epoch == 0:
            print("saving epoch model......")
            t.save(backbone_model.state_dict(), os.path.join(model_save_path, "epoch_model.pth"))


if __name__ == "__main__":
    train()