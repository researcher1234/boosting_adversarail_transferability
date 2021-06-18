import math, random, os, sys, argparse, csv, logging, shutil
from datetime import datetime
import numpy as np
import scipy.stats as st
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
from networks.resnet_feature_noise import resnet34 as resnet34_feature_noise
from networks.resnet_feature_noise import resnet50 as resnet50_feature_noise
from networks.resnet_feature_noise import resnet152 as resnet152_feature_noise
from networks.densenet_feature_noise import densenet121 as densenet121_feature_noise
from networks.densenet_feature_noise import densenet201 as densenet201_feature_noise
from utils import one_hot, load_ground_truth, Normalize, gkern, DI, get_logger, get_timestamp

def parse_arguments():
    parser = argparse.ArgumentParser(description='Trains a UAP')
    parser.add_argument('--source-model', nargs="+", default=['resnet50'])
    parser.add_argument("--target-model", nargs="+", default=['inception_v3', 'resnet50', 'densenet121', 'vgg16_bn'])
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--sampling-frequency', type=int, default=1, help='sampling the images to save the exploration time')
    parser.add_argument('--img-size', type=int, default=299, help='Image size (default: 299)')
    parser.add_argument('--log-interval', type=int, default=2, help='Logging interval (default: 2)')
    # Attack specifics
    parser.add_argument('--max-iterations', type=int, default=20, help='Maximum Iterations (default: 20)')
    parser.add_argument('--lr', type=eval, default=2./255., help='Learning rate (default: 2/255)')
    parser.add_argument('--epsilon', type=float, default=16, help='Epsilon (default: 16)')
    parser.add_argument('--loss-fn', type=str, default='ce-targeted', help='Loss function (default: ce-targeted)')
    parser.add_argument('--alpha', type=float, default=1.0, help='Loss Weight if applicable (default: 1.0)')
    parser.add_argument('--variant', type=str, default="mi-ti", help='FGSM variant (default: mi-ti)')
    parser.add_argument('--di', type=eval, default="True", choices=[True, False], help='Use DI (default: True)')
    parser.add_argument('--input-noise', default=None, help='Add noise to the adversarial example (default: None)')
    parser.add_argument('--input-noise-std', type=float, default=0.06, help='Std of the input noise (default:0.06)')
    parser.add_argument('--feature-noise-std', type=float, default=0.0, help='Feature Noise std')
    # Result Path
    parser.add_argument('--subfolder', type=str, default='development', help='Folder to store results in (default: subfolder)')
    parser.add_argument('--postfix', type=str, default='', help='Postfix to append to result path (default: \"\")')
    args = parser.parse_args()

    assert args.max_iterations % args.log_interval == 0
    return args

def main():
    
    args = parse_arguments()
    
    result_path = './results/{}/{}{}'.format(args.subfolder, get_timestamp(), args.postfix)
    os.makedirs(result_path)

    logger = get_logger(result_path)

    state = {k: v for k, v in args._get_kwargs()}
    for key, value in state.items():
        logger.info('{} : {}'.format(key, value))

    # values are standard normalization for ImageNet images, 
    # from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    num_classes = 1000

    trn = transforms.Compose([
        transforms.ToTensor(),])

    device = torch.device("cuda:0")

    # Source models
    source_model_names = args.source_model
    num_source_models = len(source_model_names)
    
    source_models = []
    for model_name in source_model_names:
        if model_name == "inception_v3":
            source_model = models.__dict__[model_name](pretrained=True, transform_input=True).eval()
        elif model_name == 'resnet34_feature_noise':
            source_model = resnet34_feature_noise(pretrained=True, noise_std=args.feature_noise_std).eval()
        elif model_name == 'resnet50_feature_noise':
            source_model = resnet50_feature_noise(pretrained=True, noise_std=args.feature_noise_std).eval()
        elif model_name == 'resnet152_feature_noise':
            source_model = resnet152_feature_noise(pretrained=True, noise_std=args.feature_noise_std).eval()
        elif model_name == 'densenet121_feature_noise':
            source_model = densenet121_feature_noise(pretrained=True, std=args.feature_noise_std).eval()
        elif model_name == 'densenet201_feature_noise':
            source_model = densenet201_feature_noise(pretrained=True, std=args.feature_noise_std).eval()
        else:
            source_model = models.__dict__[model_name](pretrained=True).eval()

        for param in source_model.parameters():
            param.requires_grad=False

        source_model.to(device)
        source_models.append(source_model)

    # Target models
    target_model_names = args.target_model
    num_target_models = len(target_model_names)

    target_models = []
    for model_name in target_model_names:
        if model_name == "inception_v3":
            target_model = models.__dict__[model_name](pretrained=True, transform_input=True).eval()
        else:
            target_model = models.__dict__[model_name](pretrained=True).eval()

        for param in target_model.parameters():
            param.requires_grad=False

        target_model.to(device)
        target_models.append(target_model)

    seed_num=1
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.backends.cudnn.deterministic = True

    channels=3
    kernel_size=5
    kernel = gkern(kernel_size, 3).astype(np.float32)
    gaussian_kernel = np.stack([kernel, kernel, kernel])
    gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
    gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()
    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, groups=channels, bias=False,padding=7)
    gaussian_filter.weight.data = gaussian_kernel

    # Hyperparameters
    image_id_list, label_ori_list, label_tar_list = load_ground_truth(os.path.join('./', 'images.csv'))
    input_path = './images/'
    # image_id_list = list(filter(lambda x: '.png' in x, os.listdir(input_path)))
    image_id_list = [x + '.png' for x in image_id_list]
    sampling_frequency = args.sampling_frequency
    image_id_list = image_id_list[0:1000:sampling_frequency]
    num_samples = len(image_id_list)
    num_batches = np.int(np.ceil(len(image_id_list)/args.batch_size))

    pos_tar=np.zeros((num_target_models, args.max_iterations//args.log_interval))
    pos_ori=np.zeros((num_target_models, args.max_iterations//args.log_interval))

    for k in range(0,num_batches):
        print(k, num_batches)
        batch_size_cur=min(args.batch_size,len(image_id_list)-k*args.batch_size)        
        X_ori = torch.zeros(batch_size_cur,3,args.img_size,args.img_size).to(device)
        delta= torch.zeros_like(X_ori,requires_grad=True).to(device)
        for i in range(batch_size_cur):          
            X_ori[i]=trn(Image.open(input_path+image_id_list[k*args.batch_size+i]))

        labels_tar=torch.tensor(label_tar_list[k*args.batch_size*sampling_frequency:(k*args.batch_size+batch_size_cur)*sampling_frequency:sampling_frequency]).to(device)
        labels_gt=torch.tensor(label_ori_list[k*args.batch_size*sampling_frequency:(k*args.batch_size+batch_size_cur)*sampling_frequency:sampling_frequency]).to(device)
        
        grad_momentum=0
        prev = float('inf')
        for t in range(args.max_iterations):
            if args.di:
                X_adv = DI(X_ori+delta)
            else:
                X_adv = X_ori+delta

            if args.input_noise == 'uniform':
                # X_adv = weight * X_adv
                pert_noise = torch.zeros_like(X_adv).uniform_(-args.input_noise_std, args.input_noise_std)
                X_adv = torch.clamp(X_adv + pert_noise, 0, 1)

            logits = 0
            for source_model in source_models:
                logits += source_model(norm(X_adv))
            logits /= num_source_models
            
            if args.loss_fn == "ce-untargeted":
                loss = -nn.CrossEntropyLoss()(logits,labels_gt)
            elif args.loss_fn == "ce-targeted":
                loss = nn.CrossEntropyLoss()(logits,labels_tar)
            elif args.loss_fn == "push-pull":
                ce_target = nn.CrossEntropyLoss()(logits, labels_tar)
                ce_gt = nn.CrossEntropyLoss()(logits, labels_gt)
                loss = ce_target - args.alpha * ce_gt
            else:
                raise ValueError
            loss.backward()

            # MI + TI
            if args.variant == 'vanilla':
                grad_a = delta.grad.clone()
            elif args.variant == "mi":
                grad_c=delta.grad.clone()
                grad_a=grad_c / torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True)+1*grad_momentum
                grad_momentum=grad_a
            elif args.variant == 'ti':
                grad_c=delta.grad.clone()
                grad_c=F.conv2d(grad_c, gaussian_kernel, bias=None, stride=1, padding=(2,2), groups=3) 
                grad_a=grad_c
            elif args.variant == 'mi-ti':
                grad_c=delta.grad.clone()
                grad_c=F.conv2d(grad_c, gaussian_kernel, bias=None, stride=1, padding=(2,2), groups=3) 
                grad_a=grad_c / torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True)+1*grad_momentum
                grad_momentum=grad_a
            else:
                raise ValueError
            delta.grad.zero_()

            delta.data=delta.data-args.lr* torch.sign(grad_a)
            delta.data=delta.data.clamp(-args.epsilon/255,args.epsilon/255)
            
            delta.data=((X_ori+delta.data).clamp(0,1))-X_ori
            
            if t % args.log_interval == (args.log_interval - 1):
                for model_idx in range(num_target_models):
                    pos_ori[model_idx, t // args.log_interval]=pos_ori[model_idx, t // args.log_interval]+torch.sum(torch.argmax(target_models[model_idx](norm(X_ori+delta)),dim=1)==labels_gt).cpu().numpy()
                    pos_tar[model_idx, t // args.log_interval]=pos_tar[model_idx, t // args.log_interval]+torch.sum(torch.argmax(target_models[model_idx](norm(X_ori+delta)),dim=1)==labels_tar).cpu().numpy()
                    
        number_seen_samples = (k+1) * args.batch_size
        logger.info('\n-- (Batch {}/{}) (Iteration {}/{}) Untargeted Accuracy --'.format(k+1, num_batches, t+1, args.max_iterations))
        for i in range(num_target_models):
            logger.info('{} -> {}'.format(args.source_model, target_model_names[i]))
            logger.info('{}'.format(1.-pos_ori[i,:]/number_seen_samples))

        logger.info('\n--  (Batch {}/{}) (Iteration {}/{}) Targeted Accuracy --'.format(k+1, num_batches, t+1, args.max_iterations))
        for i in range(num_target_models):
            logger.info('{} -> {}'.format(args.source_model, target_model_names[i]))
            logger.info('{}'.format(pos_tar[i,:]/number_seen_samples))

    torch.cuda.empty_cache()

    # Results
    logger.info('\n-- Untargeted Accuracy --')
    for i in range(num_target_models):
        logger.info('{} -> {}'.format(args.source_model, target_model_names[i]))
        logger.info('{}'.format(1.-pos_ori[i,:]/num_samples))
    
    logger.info('\n-- Targeted Accuracy --')
    for i in range(num_target_models):
        logger.info('{} -> {}'.format(args.source_model, target_model_names[i]))
        logger.info('{}'.format(pos_tar[i,:]/num_samples))

if __name__ == '__main__':
    main()