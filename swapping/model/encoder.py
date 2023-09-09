import torch
import torch.nn as nn

class MultilevelAttributesEncoder(nn.Module):
    def __init__(self):
        super(MultilevelAttributesEncoder, self).__init__()
        self.Encoder_channel = [3, 32, 64, 128, 256, 512, 1024, 1024]
        self.Encoder = nn.ModuleDict({f'layer_{i}' : nn.Sequential(
                nn.Conv2d(self.Encoder_channel[i], self.Encoder_channel[i+1], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(self.Encoder_channel[i+1]),
                nn.LeakyReLU(0.1)
            )for i in range(7)})

        self.Decoder_inchannel = [1024, 2048, 1024, 512, 256, 128]
        self.Decoder_outchannel = [1024, 512, 256, 128, 64, 32]
        self.Decoder = nn.ModuleDict({f'layer_{i}' : nn.Sequential(
                nn.ConvTranspose2d(self.Decoder_inchannel[i], self.Decoder_outchannel[i], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(self.Decoder_outchannel[i]),
                nn.LeakyReLU(0.1)
            )for i in range(6)})

        self.Upsample = nn.Upsample(scale_factor=2,align_corners=True,mode='bilinear')

    def forward(self, x):
        arr_x = []
        for i in range(7):
            x = self.Encoder[f'layer_{i}'](x)
            arr_x.append(x)


        arr_y = []
        arr_y.append(arr_x[6])
        y = arr_x[6]
        for i in range(6):
            y = self.Decoder[f'layer_{i}'](y)
            y = torch.cat((y, arr_x[5-i]), 1)
            arr_y.append(y)

        arr_y.append(self.Upsample(y))

        return arr_y