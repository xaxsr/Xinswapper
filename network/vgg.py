import torch
import torchvision.models as models

class vgg19_features(torch.nn.Module):
    def __init__(self, selected_layers, range_norm=False):
        super(vgg19_features, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True).features
        self.vgg19.eval()

        self.range_norm = range_norm
        self.selected_layers = selected_layers
        self.layer_mapping = {
            '0': 'conv1_1',
            '1': 'relu1_1',
            '2': 'conv1_2',
            '3': 'relu1_2',
            '4': 'pool1',
            '5': 'conv2_1',
            '6': 'relu2_1',
            '7': 'conv2_2',
            '8': 'relu2_2',
            '9': 'pool2',
            '10': 'conv3_1',
            '11': 'relu3_1',
            '12': 'conv3_2',
            '13': 'relu3_2',
            '14': 'conv3_3',
            '15': 'relu3_3',
            '16': 'conv3_4',
            '17': 'relu3_4',
            '18': 'pool3',
            '19': 'conv4_1',
            '20': 'relu4_1',
            '21': 'conv4_2',
            '22': 'relu4_2',
            '23': 'conv4_3',
            '24': 'relu4_3',
            '25': 'conv4_4',
            '26': 'relu4_4',
            '27': 'pool4',
            '28': 'conv5_1',
            '29': 'relu5_1',
            '30': 'conv5_2',
            '31': 'relu5_2',
            '32': 'conv5_3',
            '33': 'relu5_3',
            '34': 'conv5_4',
            '35': 'relu5_4',
            '36': 'pool5'
        }

        vgg_mean = torch.tensor([0.485, 0.456, 0.406], requires_grad=False).view(1, 3, 1, 1)
        vgg_std = torch.tensor([0.229, 0.224, 0.225], requires_grad=False).view(1, 3, 1, 1)
        self.register_buffer('vgg_mean', vgg_mean.to('cuda'))
        self.register_buffer('vgg_std', vgg_std.to('cuda'))

        for param in self.vgg19.parameters():
            param.requires_grad = False

    def normalize(self, x):
        if self.range_norm:
            x = (x + 1) / 2
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = (x - self.vgg_mean) / self.vgg_std
        return x

    def forward(self, input):
        x = self.normalize(input)
        features = {}
        for name, layer in self.vgg19._modules.items():
            x = layer(x)
            layer_name = self.layer_mapping[name]
            if layer_name in self.selected_layers:
                features[layer_name] = x
        return features