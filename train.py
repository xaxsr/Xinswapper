# Xinswapper training <xaxsr@proton.me>
import os
import cv2
import time
import torch
import numpy as np
import torch.nn.functional as F
import torch.utils.tensorboard as tensorboard
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from network.generator import xinswapper_generator
from network.discriminator import ProjectedDiscriminator
from util.data import FaceDataset

from network.vgg import vgg19_features
from util.func import tensor2im, process_latent, cosine_distance, gram_matrix, detect_landmarks

from AdaptiveWingLoss.core import models as models_aw

def main(args):
    device = 'cuda'

    output_dir = os.path.join(args.checkpoint_path, args.name)
    if not os.path.exists(output_dir):
        if not os.path.exists(args.checkpoint_path):
            os.mkdir(args.checkpoint_path)
        os.mkdir(output_dir)

    sample_dir = os.path.join(args.checkpoint_path, args.name, "samples")
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    log_dir = os.path.join(args.checkpoint_path, args.name, "logs")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    for f in os.listdir(sample_dir):
        os.remove(os.path.join(sample_dir, f))

    net_g_path = os.path.join(output_dir, f"{args.name}_netG.pth")
    net_d_path = os.path.join(output_dir, f"{args.name}_netD.pth")

    net_g = xinswapper_generator(num_style_blocks=6)
    net_g = net_g.to(device)

    if args.sample_compare:
        net_ins = xinswapper_generator(num_style_blocks=6)
        net_ins = net_ins.to(device)
        net_ins.eval()

    net_d = ProjectedDiscriminator(diffaug=False, interp224=False, **{}).to(device)
    net_d = net_d.to(device)

    if args.pretrained:
        if os.path.isfile(args.inswapper_path):
            print(f">load pretrained inswapper weights '{args.inswapper_path}'")
            net_g.load_state_dict(torch.load(args.inswapper_path, map_location='cpu'), strict=False)
            net_ins.load_state_dict(torch.load(args.inswapper_path, map_location='cpu'),strict=False)
            print(">inswapper weights successfully loaded")
        else:
            print(f">inswapper weights not found at '{args.inswapper_path}'")
            return

    if os.path.isfile(args.recognition_model_path):
        print(f">load recognition model '{args.recognition_model_path}'")
        recognition_model = torch.load(args.recognition_model_path).to(device)
        recognition_model.eval()
        print(">recognition model successfully loaded")
    else:
        print(f">recognition model not found at '{args.recognition_model_path}'")
        return

    if args.eye_landmarks:
        if os.path.isfile(args.fan_path):
            net_fan = models_aw.FAN(4, "False", "False", 98)
            checkpoint = torch.load(args.fan_path)
            if 'state_dict' not in checkpoint:
                net_fan.load_state_dict(checkpoint)
            else:
                pretrained_weights = checkpoint['state_dict']
                model_weights = net_fan.state_dict()
                pretrained_weights = {k: v for k, v in pretrained_weights.items() \
                                        if k in model_weights}
                model_weights.update(pretrained_weights)
                net_fan.load_state_dict(model_weights)
            net_fan = net_fan.to(device)
            net_fan.eval()
            del checkpoint
        else:
            print(f">AdaptiveWingLoss FAN weights not found at '{args.fan_path}'")
            return

    optimizer_g = torch.optim.AdamW(net_g.parameters(), lr=args.lr_g, betas=(args.beta1, 0.99), eps=1e-8)
    optimizer_d = torch.optim.AdamW(net_d.parameters(), lr=args.lr_d, betas=(args.beta1, 0.99), eps=1e-8)

    if args.use_scheduler:
        scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
        scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=args.scheduler_step, gamma=args.scheduler_gamma)

    if args.resume and os.path.exists(net_g_path):
        print(">loading saved checkpoint G...")
        checkpoint = torch.load(net_g_path)
        net_g.load_state_dict(checkpoint['model_state_dict'])
        if not args.reset:
            optimizer_g.load_state_dict(checkpoint['optimizer_state_dict'])
        del checkpoint

    if args.resume and os.path.exists(net_d_path):
        print(">loading saved checkpoint D...")
        checkpoint = torch.load(net_d_path)
        net_d.load_state_dict(checkpoint['model_state_dict'])
        if not args.reset:
            optimizer_d.load_state_dict(checkpoint['optimizer_state_dict'])
        del checkpoint

    dataset = FaceDataset(args.data_root, args.image_size, same_prob=args.same_prob, same_id=args.same_id, regularization_path=args.data_reg_root)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)

    id_criterion = torch.nn.L1Loss()
    style_criterion = torch.nn.L1Loss()
    feat_criterion = torch.nn.L1Loss()
    eyes_criterion = torch.nn.MSELoss()

    style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    vgg19 = vgg19_features(style_layers).to(device)

    logger = tensorboard.SummaryWriter(log_dir)

    step = 0
    tstart = time.time()

    for epoch in range(args.epoch_max):
        for data in dataloader:
            net_g.train()
            net_d.train()

            step += 1
            target, source, source_id, flags = data

            target = target.to(device)
            source = source.to(device)
            source_id = source_id.to(device)
            flags = flags.to(device)

            with torch.no_grad():
                latent =  process_latent(recognition_model, source_id)

            different_id = torch.ones_like(flags)
            optimizer_g.zero_grad()

            img_fake = net_g(target, latent)

            with torch.no_grad():
                img_fake_ins = net_ins(target, latent)

            logits, fake_features = net_d(img_fake, None)

            fake_denorm = (img_fake + 1) / 2
            latent_fake = process_latent(recognition_model, fake_denorm)

            norm_target =  v2.functional.normalize(target, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False)
            norm_source =  v2.functional.normalize(source, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False)

            # adversarial loss
            adversarial_loss = torch.sum(torch.nn.ReLU()(1.0 - logits).mean() * different_id) / (different_id.sum() + 1e-4)

            # eye landmark loss
            if args.eye_landmarks:
                x_eyes, x_heatmap_left, x_heatmap_right = detect_landmarks(net_fan, target, args.image_size)
                y_eyes, y_heatmap_left, y_heatmap_right = detect_landmarks(net_fan, img_fake, args.image_size)
                eyes_loss = eyes_criterion(x_heatmap_left, y_heatmap_left) + eyes_criterion(x_heatmap_right, y_heatmap_right)

            # id loss
            latent = F.normalize(latent, p=2, dim=1)
            latent_fake = F.normalize(latent_fake, p=2, dim=1)
            inner_product = (torch.bmm(latent.unsqueeze(1),latent_fake.unsqueeze(2)).squeeze())
            id_loss = id_criterion(torch.ones_like(inner_product), inner_product)
            fake_to_source_cosine = cosine_distance(latent, latent_fake).mean()
            id_loss = id_loss + fake_to_source_cosine

            # reconstruction loss
            reconstruction_loss = torch.sum(0.5 * torch.mean(torch.pow(img_fake - norm_target, 2).reshape(args.batch_size, -1), dim=1) * flags) / (flags.sum() + 1e-6)

            # style loss
            style_loss = 0.0
            swapped_features = vgg19(img_fake)
            with torch.no_grad():
                target_features = vgg19(norm_target)
            for layer in style_layers:
                gram_target = gram_matrix(target_features[layer])
                gram_swapped = gram_matrix(swapped_features[layer])
                style_loss += style_criterion(gram_swapped, gram_target)

            # feature loss
            real_features = net_d.get_feature(norm_target)
            feature_loss = feat_criterion(fake_features["3"], real_features["3"])  # 0.0

            g_loss = args.weights_adv*adversarial_loss + args.weights_rec*reconstruction_loss + args.weights_id*id_loss + args.weights_style*style_loss + args.weights_feat*feature_loss + args.weights_eyes*eyes_loss

            g_loss.backward()
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(net_g.parameters(), max_norm=1.0)
            optimizer_g.step()
            if args.use_scheduler:
                scheduler_g.step()

            optimizer_d.zero_grad()

            d_out_real, _ = net_d(norm_source, None)
            score_real = torch.sum(torch.nn.ReLU()(1.0 - d_out_real).mean() * flags) / (flags.sum() + 1e-4)

            d_out_fake, _ = net_d(img_fake.detach(), None)
            score_fake = torch.sum(torch.nn.ReLU()(1.0 + d_out_fake).mean() * flags) / (flags.sum() + 1e-4)

            d_loss = 0.5 * (score_real.mean() + score_fake.mean())

            d_loss.backward()
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(net_d.parameters(), max_norm=1.0)
            optimizer_d.step()
            if args.use_scheduler:
                scheduler_d.step()

            net_g.eval()
            net_d.eval()

            if step % args.log_steps == 0:
                telapsed = time.time() - tstart

                losses = {
                    "g_loss": g_loss.item(),
                    "g_adversarial": adversarial_loss.item(),
                    "g_id": id_loss.item(),
                    "g_reconstruction": reconstruction_loss.item(),
                    "g_eyes": eyes_loss.item(),
                    "g_style": style_loss.item(),
                    "g_feature": feature_loss.item(),
                    "d_fake": score_fake.item(),
                    "d_real": score_real.item(),
                    "d_loss": d_loss.item(),
                }

                for tag, value in losses.items():
                    logger.add_scalar(tag, value, step)

                message = '(epoch: %d\t iters: %d\t time: %.5f)\t' % (epoch, step, telapsed)
                for k, v in losses.items():
                    message += f'{k}: {v:.6g}\t'
                    #'%s: %.5f\t' % (k, v)
                print(message)

            if step % args.sample_steps == 0:
                img_stack = []
                for x in range(args.batch_size):
                    if args.sample_compare:
                        img_stack.append(np.hstack([
                            tensor2im(source[x], text='source'),
                            tensor2im(target[x], text='target'),
                            tensor2im(img_fake[x], range_norm=True, text='Xinswapper'),
                            tensor2im(img_fake_ins[x], range_norm=True, text='inswapper')
                        ]))
                    else:
                        img_stack.append(np.hstack([
                            tensor2im(source[x], text='source'),
                            tensor2im(target[x], text='target'),
                            tensor2im(img_fake[x], range_norm=True, text='Xinswapper'),
                        ]))

                n = 0
                img_out = os.path.join(sample_dir, "%05d.jpg" % n)
                while os.path.exists(img_out):
                    n+=1
                    img_out = os.path.join(sample_dir, "%05d.jpg" % n)
                cv2.imwrite(img_out, np.vstack(img_stack))

            if step % args.checkpoint_steps == 0:
                print(">saving checkpoint...")

                if os.path.exists(net_g_path):
                    os.rename(net_g_path, net_g_path + '.bak')
                if os.path.exists(net_d_path):
                    os.rename(net_d_path, net_d_path + '.bak')

                torch.save({
                    'model_state_dict': net_g.state_dict(),
                    'optimizer_state_dict': optimizer_g.state_dict()
                }, net_g_path)

                torch.save({
                    'model_state_dict': net_d.state_dict(),
                    'optimizer_state_dict': optimizer_d.state_dict()
                }, net_d_path)

                if os.path.exists(net_g_path + '.bak'):
                    os.remove(net_g_path + '.bak')
                if os.path.exists(net_d_path + '.bak'):
                    os.remove(net_d_path + '.bak')

                print(">checkpoint saved!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='my_run',  help='name of run')
    parser.add_argument('--reset', action="store_true", help='reset optimizer')
    parser.add_argument('--resume', action="store_true", help='resume training')
    parser.add_argument('--data_root', type=str, default='./dataset_root/', help='dataset root folder')
    parser.add_argument('--data_reg_root', type=str, default=None, help='dataset path for regularization (tested with WebFace260) (optional/experimental)')
    parser.add_argument('--inswapper_path', type=str, default='./weights/inswapper_128.pth', help='path to inswapper weights and used if pretrained=True')
    parser.add_argument('--recognition_model_path', type=str, default='./weights/w600k_r50.pt', help='path to recognition model')
    parser.add_argument('--fan_path', type=str, default='./weights/WFLW_4HG.pth', help='path to AdaptiveWingLoss FAN weights for eye landmark loss')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/', help='path to save checkpoints')
    parser.add_argument('--log_path', type=str, default='./logs/', help='path to save logs')
    parser.add_argument('--log_steps', type=int, default=10, help='number of steps for logging')
    parser.add_argument('--sample_steps', type=int, default=10, help='number of steps to save sample')
    parser.add_argument('--sample_compare', type=bool, default=True, help='compare outputs to original inswapper model in saved samples')
    parser.add_argument('--checkpoint_steps', type=int, default=500, help='number of steps to save checkpoint')
    parser.add_argument('--pretrained', type=bool, default=True, help='initialize model with the pretrained inswapper weights')
    parser.add_argument('--clip_grad', type=bool, default=True, help='enable clip gradient')
    parser.add_argument('--same_prob', type=float, default=0.2, help='probability of batch containing source and target from same identity folder')
    parser.add_argument('--same_id', type=bool, default=False, help='different faces from the same identity folder are flagged as same')
    parser.add_argument('--eye_landmarks', type=bool, default=True, help='use AdaptiveWingLoss eye landmark detection loss')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size')
    parser.add_argument('--epoch_max', type=int, default=100000, help='maximum number of times to process dataset')
    parser.add_argument('--beta1', type=float, default=0.0, help='adam momentum term')
    parser.add_argument('--lr_g', type=float, default=2e-6, help='learning rate for generator')
    parser.add_argument('--lr_d', type=float, default=5e-6, help='learning rate for discriminator')
    parser.add_argument('--use_scheduler', type=bool, default=False, help='use step scheduler during training')
    parser.add_argument('--scheduler_step', default=5000, type=int, help="period of lr decay")
    parser.add_argument('--scheduler_gamma', default=0.2, type=float, help='multiplicative factor for lr decay')
    parser.add_argument('--image_size', type=int, default=128, help='image size')

    parser.add_argument('--weights_adv', type=float, default=1.0, help='weight for adversarial loss')
    parser.add_argument('--weights_rec', type=float, default=8.0, help='weight for reconstruction loss')
    parser.add_argument('--weights_id', type=float, default=35.0, help='weight for id loss')
    parser.add_argument('--weights_style', type=float, default=30.0, help='weight for style loss')
    parser.add_argument('--weights_feat', type=float, default=10.0, help='weight for feature loss')
    parser.add_argument('--weights_eyes', type=float, default=5.0, help='weight for eye landmark loss')

    args = parser.parse_args()

    main(args)
