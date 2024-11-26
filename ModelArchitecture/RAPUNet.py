import timm
import torch
from torch import nn
import torch.nn.functional as F
from RAPU_blocks import ResnetBlock, RAPUBlock, Convf_bn_act, SBA, same_padding

kernel_initializer = 'he_uniform'
interpolation = "nearest"

class RAPUNet(nn.Module):
    def __init__(self, in_channels, out_classes, starting_kernels):
        super().__init__()
        self.backbone = timm.create_model('caformer_s18.sail_in22k_ft_in1k_384', pretraned=True, features_only=True)


        self.p1_conv2d = nn.Conv2d(in_channels, starting_kernels * 2, 3, stride=2)
    
        #from metaformer
        self.p2_conv2d = nn.Conv2d(64, starting_kernels * 4, 1) #96,96
        self.p3_conv2d = nn.Conv2d(128, starting_kernels * 8, 1) #48, 48
        self.p4_conv2d = nn.Conv2d(320, starting_kernels * 16, 1) #24, 24
        self.p5_conv2d = nn.Conv2d(512, starting_kernels * 32, 1) #12, 12
    
        self.t0_rapu = RAPUBlock(in_channels, starting_kernels) #384, 384

        self.l1i_conv2d = nn.Conv2d(starting_kernels, starting_kernels * 2, 2, stride=2)
        self.t1_rapu = RAPUBlock(starting_kernels * 2, starting_kernels * 2) #192, 192

        self.l2i_conv2d = nn.Conv2d(starting_kernels * 2, starting_kernels * 4, 2, stride=2)
        self.t2_rapu = RAPUBlock(starting_kernels * 4, starting_kernels * 4) #96, 96

        self.l3i_conv2d = nn.Conv2d(starting_kernels * 4, starting_kernels * 8, 2, stride=2)
        self.t3_rapu = RAPUBlock(starting_kernels * 8, starting_kernels * 8) #48, 48

        self.l4i_conv2d = nn.Conv2d(starting_kernels * 8, starting_kernels * 16, 2, stride=2)
        self.t4_rapu = RAPUBlock(starting_kernels * 16, starting_kernels * 16) #24, 24

        self.l5i_conv2d = nn.Conv2d(starting_kernels * 16, starting_kernels * 32, 2, stride=2)

        self.t51_res = ResnetBlock(starting_kernels * 32, starting_kernels * 32)
        self.t52_res = ResnetBlock(starting_kernels * 32, starting_kernels * 32)
        self.t53_res = ResnetBlock(starting_kernels * 32, starting_kernels * 16)
        self.t54_res = ResnetBlock(starting_kernels * 16, starting_kernels * 16)

        #Aggregation
        kernels = 32

            #High-level feature
        self.h_conv_block = Convf_bn_act(starting_kernels * 40, kernels, 1) # concat encoder output 2 3 4 = 8 + 16 + 16 = 40
        self.h_conv = nn.Conv2d(kernels, 1, 1, bias=False)

            #Low-level feature
        self.L_conv_block = Convf_bn_act(starting_kernels * 4, kernels, 3)
        self.H_conv_block = Convf_bn_act(starting_kernels * 32, kernels, 1)
        self.SBA = SBA(kernels, kernels)

        self.final_conv = nn.Conv2d(1, out_classes, 1)

    
    def forward(self, x):
        backbone_stages = self.backbone(x)

        x_tmp = same_padding(x, ksize=3, stride=2)
        p1 = self.p1_conv2d(x_tmp)

        #from metaformer
        p2 = self.p2_conv2d(backbone_stages[0]) #96, 96
        p3 = self.p3_conv2d(backbone_stages[1]) #48, 48
        p4 = self.p4_conv2d(backbone_stages[2]) #24, 24
        p5 = self.p5_conv2d(backbone_stages[3]) #12, 12

        t0 = self.t0_rapu(x) #384, 384

        t0_tmp = same_padding(t0, ksize=2, stride=2)
        l1i = self.l1i_conv2d(t0_tmp)
        s1 = p1 + l1i
        t1 = self.t1_rapu(s1) #192, 192

        t1_tmp = same_padding(t1, ksize=2, stride=2)
        l2i = self.l2i_conv2d(t1_tmp)
        s2 = p2 + l2i
        encoder_output1 = self.t2_rapu(s2) #96, 96

        encoder_output1_tmp = same_padding(encoder_output1, ksize=2, stride=2)
        l3i = self.l3i_conv2d(encoder_output1_tmp)
        s3 = p3 + l3i
        encoder_output2 = self.t3_rapu(s3) #48, 48

        encoder_output2_tmp = same_padding(encoder_output2, ksize=2, stride=2)
        l4i = self.l4i_conv2d(encoder_output2_tmp)
        s4 = p4 + l4i
        encoder_output3 = self.t4_rapu(s4) #24, 24

        encoder_output3_tmp = same_padding(encoder_output3, ksize=2, stride=2)
        l5i = self.l5i_conv2d(encoder_output3_tmp)
        s5 = p5 + l5i
        encoder_output4 = self.t51_res(s5)
        encoder_output4 = self.t52_res(encoder_output4)
        encoder_output4 = self.t53_res(encoder_output4)
        encoder_output4 = self.t54_res(encoder_output4) #12, 12

        #Aggregation
        high_level_feature = torch.cat(
            [
                F.interpolate(encoder_output3, scale_factor=2),
                F.interpolate(encoder_output4, scale_factor=4)
            ],
            dim=1 # Concatenate along channel dimension
        )
        high_level_feature = torch.cat(
            [
                high_level_feature,
                encoder_output2
            ],
            dim=1
        )
        high_level_feature = self.h_conv_block(high_level_feature)
        high_level_feature = self.h_conv(high_level_feature) #1, 48, 48


        L_input = self.L_conv_block(encoder_output1) #32, 96, 96
        H_input = torch.cat(
            [
                encoder_output3,
                F.interpolate(encoder_output4, scale_factor=2)
            ],
            dim=1
        )
        H_input = self.H_conv_block(H_input)
        H_input = F.interpolate(H_input, scale_factor=2) #32, 48, 48
        low_level_feature = self.SBA(L_input, H_input) #1, 96, 96


        high_level_feature = F.interpolate(high_level_feature, scale_factor=8, mode="bilinear")
        low_level_feature = F.interpolate(low_level_feature, scale_factor=4, mode="nearest")

        output = low_level_feature + high_level_feature
        output = self.final_conv(output)
        output = torch.sigmoid(output)

        return output