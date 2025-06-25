import torch
from torch import nn


def weights_init_normal(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def discriminator_block(in_filters, out_filters, kernel_size, stride, padding, bn=True):
    block = [nn.Conv2d(in_filters, out_filters, kernel_size, stride, padding),
             nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
    if bn:
        block.append(nn.BatchNorm2d(out_filters, 0.8))
    return block


class Discriminator(nn.Module):
    def __init__(self, large=False):
        super().__init__()

        self.model = nn.Sequential(
            *discriminator_block(128, 64, 3, 2 if large else 1, 1, bn=False),
            *discriminator_block(64, 32, 3, 2, 1),
            *discriminator_block(32, 16, 3, 2 if large else 1, 1),
            *discriminator_block(16, 8, 3, 2 if large else 1, 1),
        )
        # h, w = hw
        # for _ in range(4):
        #     h = (h + 2 * padding - kernel_size) // stride + 1
        #     w = (w + 2 * padding - kernel_size) // stride + 1
        self.adv_layer = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(1)
        )
        self.apply(weights_init_normal)

    def forward(self, img):  # torch.Size([4, 128, 240, 427])
        out = self.model(img)  # torch.Size([4, 8, 15, 27])
        # out = out.reshape(out.shape[0], -1)  # torch.Size([4, 2520=15*21])
        validity = self.adv_layer(out)
        return validity
