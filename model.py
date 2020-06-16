import torch as t
from torch import nn


class InvertBlock(nn.Module):

    def __init__(self, in_channels, expand_ratio, stride, out_channels):
        super(InvertBlock, self).__init__()
        middle_layer_channels = expand_ratio * in_channels
        self.expand_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=middle_layer_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=middle_layer_channels),
            nn.ReLU()
        )
        self.deepwise_conv = nn.Sequential(
            nn.Conv2d(in_channels=middle_layer_channels, out_channels=middle_layer_channels, kernel_size=3, stride=stride, padding=1, groups=middle_layer_channels, bias=False),
            nn.BatchNorm2d(num_features=middle_layer_channels),
            nn.ReLU()
        )
        self.shrink_block = nn.Sequential(
            nn.Conv2d(in_channels=middle_layer_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_channels)
        )

    def forward(self, x):
        input_value = x
        x = self.expand_block(x)
        x = self.deepwise_conv(x)
        x = self.shrink_block(x)
        if x.size() == input_value.size():
            return input_value + x
        return x


class BackBone(nn.Module):

    def __init__(self):
        super(BackBone, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        ) # 64 * 56 * 56
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )  # 64 * 56 * 56
        self.conv3_1 = InvertBlock(in_channels=64, expand_ratio=2, stride=2, out_channels=64)  # 64 * 28 * 28
        self.conv3_2 = InvertBlock(in_channels=64, expand_ratio=2, stride=1, out_channels=64)  # 64 * 28 * 28
        self.conv3_3 = InvertBlock(in_channels=64, expand_ratio=2, stride=1, out_channels=64)  # 64 * 28 * 28
        self.conv3_4 = InvertBlock(in_channels=64, expand_ratio=2, stride=1, out_channels=64)  # 64 * 28 * 28
        self.conv3_5 = InvertBlock(in_channels=64, expand_ratio=2, stride=1, out_channels=64)  # 64 * 28 * 28
        self.conv4_1 = InvertBlock(in_channels=64, expand_ratio=2, stride=2, out_channels=128)  # 128 * 14 * 14
        self.conv5_1 = InvertBlock(in_channels=128, expand_ratio=4, stride=1, out_channels=128)  # 128 * 14 * 14
        self.conv5_2 = InvertBlock(in_channels=128, expand_ratio=4, stride=1, out_channels=128)  # 128 * 14 * 14
        self.conv5_3 = InvertBlock(in_channels=128, expand_ratio=4, stride=1, out_channels=128)  # 128 * 14 * 14
        self.conv5_4 = InvertBlock(in_channels=128, expand_ratio=4, stride=1, out_channels=128)  # 128 * 14 * 14
        self.conv5_5 = InvertBlock(in_channels=128, expand_ratio=4, stride=1, out_channels=128)  # 128 * 14 * 14
        self.conv5_6 = InvertBlock(in_channels=128, expand_ratio=4, stride=1, out_channels=128)  # 128 * 14 * 14
        self.S1 = InvertBlock(in_channels=128, expand_ratio=2, stride=1, out_channels=16)  # 16 * 14 * 14
        self.S2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU()
        )  # 32 * 7 * 7
        self.S3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=128)
        )  # 128 * 1 * 1
        self.avg_pool14 = nn.AvgPool2d(kernel_size=14, stride=1, padding=0)
        self.avg_pool7 = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.result = nn.Linear(in_features=16 + 32 + 128, out_features=196)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)
        x = self.conv3_5(x)
        aux_input_feature = x
        x = self.conv4_1(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.conv5_4(x)
        x = self.conv5_5(x)
        x = self.conv5_6(x)
        x = self.S1(x)
        S1 = self.avg_pool14(x).view((x.size()[0], -1))
        x = self.S2(x)
        S2 = self.avg_pool7(x).view((x.size()[0], -1))
        x = self.S3(x)
        S3 = x.view((x.size()[0], -1))
        result = self.sigmoid(self.result(t.cat((S1, S2, S3), dim=1)))
        return result, aux_input_feature


class AuxNet(nn.Module):

    def __init__(self):
        super(AuxNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
        )
        self.result = nn.Sequential(
            nn.Linear(in_features=128, out_features=32, bias=False),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view((x.size()[0], -1))
        result = self.result(x)
        return result


if __name__ == "__main__":
    model = BackBone()
    aux = AuxNet()
    d = t.randn(2, 3, 112, 112)
    output = model(d)
    aux_result = aux(output[1])
    print(aux_result.size())