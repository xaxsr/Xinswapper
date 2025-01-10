import torch
import torch.nn as nn

class resnet_adain_block(nn.Module):
    def __init__(self, dim, embedding_size):
        super(resnet_adain_block, self).__init__()

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=0, bias=True)
        )
        self.style1 = nn.Linear(embedding_size, dim * 2, bias=True)

        self.activation = nn.ReLU(True)

        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=0, bias=True)
        )
        self.style2 = nn.Linear(embedding_size, dim * 2, bias=True)

    def forward(self, x, z_id):
        y = self.conv1(x)

        # mean and variance 1
        conv1_mean = torch.mean(y, dim=[2, 3], keepdim=True)
        conv1_centered = y - conv1_mean
        conv1_variance = torch.mean(conv1_centered ** 2, dim=[2, 3], keepdim=True) + 1e-08
        conv1_std = torch.sqrt(conv1_variance)
        conv1_normalized = conv1_centered * (1 / conv1_std)

        # style transform 1
        style1 = self.style1(z_id)
        style1_expanded = style1.unsqueeze(2).unsqueeze(3)
        style1_scale = style1_expanded[:, :1024, :, :]
        style1_bias = style1_expanded[:, 1024:, :, :]

        # apply
        adain1 = (conv1_normalized * style1_scale) + style1_bias
        activated = self.activation(adain1)

        y = self.conv2(activated)

        # mean and variance 2
        conv2_mean = torch.mean(y, dim=[2, 3], keepdim=True)
        conv2_centered = y - conv2_mean
        conv2_variance = torch.mean(conv2_centered ** 2, dim=[2, 3], keepdim=True) + 1e-8
        conv2_std = torch.sqrt(conv2_variance)
        conv2_normalized = conv2_centered  * (1 / conv2_std)

        # style transform 2
        style2 = self.style2(z_id)
        style2_expanded = style2.unsqueeze(2).unsqueeze(3)
        style2_scale = style2_expanded[:, :1024, :, :]
        style2_bias = style2_expanded[:, 1024:, :, :]

        # apply
        adain2 = (conv2_normalized * style2_scale) + style2_bias

        # residual
        out = x + adain2

        return out

class xinswapper_generator(nn.Module):
    def __init__(self, num_style_blocks=6):
        super(xinswapper_generator, self).__init__()

        self.emap = nn.Parameter(torch.rand((512, 512), dtype=torch.float32))

        self.num_style_blocks = num_style_blocks

        self.first_layer = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(3, 128, kernel_size=7, padding=0),
            nn.LeakyReLU(0.2)
        )
        self.encoder1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )

        style_block = [resnet_adain_block(1024, 512) for _ in range(num_style_blocks)]
        self.style_block = nn.Sequential(*style_block)

        self.decoder3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.decoder2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.last_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(128, 3, kernel_size=7, padding=0), nn.Tanh())

    def forward(self, target, source):
        x = target

        latents = []
        for i in range(source.size(0)):
            latent = source[i].unsqueeze(0)
            n_e = latent / torch.norm(latent, p=2, dim=1, keepdim=True)
            latent = n_e.reshape((1, -1))
            latent = torch.matmul(latent, self.emap)
            latent = latent / torch.norm(latent, p=2, dim=1, keepdim=True)
            latents.append(latent)
        latents = torch.cat(latents)

        x = self.first_layer(x)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)

        for i in range(self.num_style_blocks):
            x = self.style_block[i](x, latents)

        x = self.decoder3(x)
        x = self.decoder2(x)
        x = self.decoder1(x)
        output = self.last_layer(x)

        return output

class xinswapper_generator_onnx_ready(nn.Module):
    def __init__(self, num_style_blocks=6):
        super(xinswapper_generator_onnx_ready, self).__init__()

        self.num_style_blocks = num_style_blocks

        self.first_layer = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(3, 128, kernel_size=7, padding=0),
            nn.LeakyReLU(0.2)
        )
        self.encoder1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )

        style_block = [resnet_adain_block(1024, 512) for _ in range(num_style_blocks)]
        self.style_block = nn.Sequential(*style_block)

        self.decoder3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.decoder2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.last_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(128, 3, kernel_size=7, padding=0), nn.Tanh())

    def forward(self, target, source):
        x = target

        x = self.first_layer(x)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)

        for i in range(self.num_style_blocks):
            x = self.style_block[i](x, source)

        x = self.decoder3(x)
        x = self.decoder2(x)
        x = self.decoder1(x)
        output = self.last_layer(x)

        return (output + 1) / 2
