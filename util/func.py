import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from functools import reduce

def tensor2im(input_image, range_norm=False, text=None):
    image_numpy = input_image.data.cpu().float().clamp_(-1,1).numpy()
    if not range_norm:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    else:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0

    if text is not None:
        image_numpy = cv2.cvtColor(image_numpy.astype(np.uint8),cv2.COLOR_RGB2BGR)
        cv2.putText(image_numpy, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        return image_numpy.astype(np.uint8)

    return cv2.cvtColor(image_numpy,cv2.COLOR_RGB2BGR).astype(np.uint8)

def process_latent(recognition_model, x):
    latents = []
    resize = v2.Resize((112, 112), antialias=True)
    for j in range(x.size(0)):
        img = resize(x[j])
        img = img.permute(1,2,0)
        img = img[:, :, [2,1,0]]
        img = torch.mul(img, 255.0)
        img = torch.sub(img, 127.5)
        img = torch.div(img, 127.5)
        img = img.permute(2, 0, 1).unsqueeze(0) ##3,112,112
        latent = recognition_model(img)
        latents.append(latent)
    latents = torch.cat(latents).to('cuda')
    return latents

def gram_matrix(x):
    b, c, h, w = x.size()
    features = x.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2)) / (c * h * w)
    return G

def cosine_distance(vector1, vector2):
    vec1 = vector1.view(vector1.size(0), -1)
    vec2 = vector2.view(vector2.size(0), -1)
    a = (vec1 * vec2).sum(dim=1)
    b = (vec1 * vec1).sum(dim=1)
    c = (vec2 * vec2).sum(dim=1)
    cosine_similarity = a / (torch.sqrt(b) * torch.sqrt(c))
    cosine_distance = 1 - cosine_similarity
    return cosine_distance

def transform(point, center, scale, resolution, rotation=0, invert=False):
        _pt = np.ones(3)
        _pt[0] = point[0]
        _pt[1] = point[1]

        h = 200.0 * scale
        t = np.eye(3)
        t[0, 0] = resolution / h
        t[1, 1] = resolution / h
        t[0, 2] = resolution * (-center[0] / h + 0.5)
        t[1, 2] = resolution * (-center[1] / h + 0.5)

        if rotation != 0:
            rotation = -rotation
            r = np.eye(3)
            ang = rotation * math.pi / 180.0
            s = math.sin(ang)
            c = math.cos(ang)
            r[0][0] = c
            r[0][1] = -s
            r[1][0] = s
            r[1][1] = c

            t_ = np.eye(3)
            t_[0][2] = -resolution / 2.0
            t_[1][2] = -resolution / 2.0
            t_inv = torch.eye(3)
            t_inv[0][2] = resolution / 2.0
            t_inv[1][2] = resolution / 2.0
            t = reduce(np.matmul, [t_inv, r, t_, t])

        if invert:
            t = np.linalg.inv(t)
        new_point = (np.matmul(t, _pt))[0:2]

        return new_point.astype(int)

def get_preds_fromhm(hm, center=None, scale=None, rot=None):
    max, idx = torch.max(
        hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    idx += 1
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1)
    preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)

    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = torch.FloatTensor(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j].add_(diff.sign_().mul_(.25))

    preds.add_(-0.5)

    preds_orig = torch.zeros(preds.size())
    if center is not None and scale is not None:
        for i in range(hm.size(0)):
            for j in range(hm.size(1)):
                preds_orig[i, j] = transform(
                    preds[i, j], center, scale, hm.size(2), rot, True)

    return preds, preds_orig

def detect_landmarks(model_ft, inputs, size=128):
    inputs = F.interpolate(inputs, size=(256,256), mode='bicubic')

    outputs, boundary_channels = model_ft(inputs)
    pred_heatmap = outputs[-1][:, :-1, :, :].cpu()
    pred_landmarks, _ = get_preds_fromhm(pred_heatmap)
    landmarks = pred_landmarks*4.0
    eyes = torch.cat((landmarks[:,96,:], landmarks[:,97,:]), 1)

    heatmap_1 = pred_heatmap[:, 96, :, :]
    heatmap_2 = pred_heatmap[:, 97, :, :]

    heatmap_1_resized = F.interpolate(heatmap_1.unsqueeze(0),
                                      size=[int(heatmap_1.size(2) // (256 / size)), int(heatmap_1.size(1) // (256 / size))], mode='bilinear',
                                      align_corners=False).squeeze(1)
    heatmap_2_resized = F.interpolate(heatmap_2.unsqueeze(0),
                                      size=[int(heatmap_2.size(2) // (256 / size)), int(heatmap_2.size(1) // (256 / size))], mode='bilinear',
                                      align_corners=False).squeeze(1)

    return eyes // (256 / size), heatmap_1_resized, heatmap_2_resized
