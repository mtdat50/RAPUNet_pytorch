import torch
import torch.nn as nn
import torch.nn.functional as F

def same_padding(x, ksize=1, stride=1, dilation=1):
    """
    Same padding for 2D convolution.
    """
    in_height, in_width = x.shape[-2:] # assumes NCHW format
    out_height = (in_height + stride - 1) // stride
    out_width = (in_width + stride - 1) // stride

    effective_ksize_h = ksize + (ksize - 1) * (dilation - 1)
    effective_ksize_w = ksize + (ksize - 1) * (dilation - 1)

    pad_h = max(0, (out_height - 1) * stride + effective_ksize_h - in_height)
    pad_w = max(0, (out_width - 1) * stride + effective_ksize_w - in_width)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
    return x


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, kernels, dilation_rate = 1):
        super().__init__()
        self.dilation_rate = dilation_rate
        self.conv1 = nn.Conv2d(in_channels, kernels, kernel_size=1, dilation=dilation_rate)

        self.conv2 = nn.Conv2d(in_channels, kernels, kernel_size=3, dilation=dilation_rate)
        self.batchnorm1 = nn.BatchNorm2d(kernels)

        self.conv3 = nn.Conv2d(kernels, kernels, kernel_size=3, dilation=dilation_rate)
        self.batchnorm2 = nn.BatchNorm2d(kernels)

        self.batchnorm_final = nn.BatchNorm2d(kernels)

    def forward(self, x):
        x = same_padding(x, dilation=self.dilation_rate)
        x1 = self.conv1(x)
        x1 = torch.relu(x1)

        x2 = same_padding(x, ksize=3, dilation=self.dilation_rate)
        x2 = self.conv2(x2)
        x2 = torch.relu(x2)
        x2 = self.batchnorm1(x2)

        x2 = same_padding(x2, ksize=3, dilation=self.dilation_rate)
        x2 = self.conv3(x2)
        x2 = torch.relu(x2)
        x2 = self.batchnorm2(x2)

        x_final = x1 + x2
        x_final = self.batchnorm_final(x_final)
        return x_final


class AtrousBlock(nn.Module):
    def __init__(self, in_channels, kernels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, kernels, kernel_size=3, dilation=1)
        self.batchnorm1 = nn.BatchNorm2d(kernels)

        self.conv2 = nn.Conv2d(kernels, kernels, kernel_size=3, dilation=2)
        self.batchnorm2 = nn.BatchNorm2d(kernels)

        self.conv3 = nn.Conv2d(kernels, kernels, kernel_size=3, dilation=3)
        self.batchnorm3 = nn.BatchNorm2d(kernels)

    def forward(self, x):
        x = same_padding(x, ksize=3)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.batchnorm1(x)

        x = same_padding(x, ksize=3, dilation=2)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.batchnorm2(x)

        x = same_padding(x, ksize=3, dilation=3)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.batchnorm3(x)

        return x


class RAPUBlock(nn.Module):
    def __init__(self, in_channels, kernels):
        super().__init__()
        self.batchnorm1 = nn.BatchNorm2d(in_channels)
        self.atrous_block = AtrousBlock(in_channels, kernels)
        self.resnet_block = ResnetBlock(in_channels, kernels)
        self.batchnorm2 = nn.BatchNorm2d(kernels)

    def forward(self, x):
        x = self.batchnorm1(x)
        x1 = self.atrous_block(x)
        x2 = self.resnet_block(x)
        x_final = x1 + x2
        x_final = self.batchnorm2(x_final)
        return x_final



class Convf_bn_act(nn.Module):
    def __init__(self, in_channels, kernels, kernel_size, stride=1, activation='relu'):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, kernels, kernel_size=kernel_size, stride=stride, bias=False)
        self.batchnorm = nn.BatchNorm2d(kernels)
        self.activation = activation
    
    def forward(self, x):
        x = same_padding(x, ksize=self.kernel_size, stride=self.stride)
        x = self.conv(x)
        x = self.batchnorm(x)

        if self.activation == 'relu':
            x = torch.relu(x)
        elif self.activation == 'sigmoid':
            x = torch.sigmoid(x)

        return x


class SBA(nn.Module):
    def __init__(self, L_in_channels, H_in_channels, return_feature = False):
        super().__init__()
        kernels = 16
        self.return_feature = return_feature

        self.L_conv1 = nn.Conv2d(L_in_channels, kernels, kernel_size=1, bias=False)
        self.H_conv1 = nn.Conv2d(H_in_channels, kernels, kernel_size=1, bias=False)

        self.L_conv_block = Convf_bn_act(kernels, kernels, 1)
        self.H_conv_block = Convf_bn_act(kernels, kernels, 1)

        self.final_conv_block = Convf_bn_act(kernels * 2, kernels * 2, 3)
        self.output_conv = nn.Conv2d(kernels * 2, 1, kernel_size=1, bias=False)

    def forward(self, L_input, H_input):
        L_input = self.L_conv1(L_input)
        H_input = self.H_conv1(H_input)

        g_L = torch.sigmoid(L_input)
        g_H = torch.sigmoid(H_input)

        L_input = self.L_conv_block(L_input)
        H_input = self.H_conv_block(H_input)

        L_resized = F.interpolate(g_H * H_input, scale_factor=2, mode='nearest')
        H_resized = F.interpolate(g_L * L_input, size=(H_input.shape[2], H_input.shape[3]), mode='bilinear')

        L_feature = L_input + L_input * g_L + (1 - g_L) * L_resized
        H_feature = H_input + H_input * g_H + (1 - g_H) * H_resized

        H_feature = F.interpolate(H_feature, scale_factor=2, mode='nearest')
        combined_feature = torch.cat([L_feature, H_feature], dim=1)  # Concatenate along channel dimension

        combined_feature = self.final_conv_block(combined_feature)
        out = self.output_conv(combined_feature)
        
        if self.return_feature:
            return out, combined_feature
        return out
