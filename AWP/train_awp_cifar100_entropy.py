import argparse
import logging
import sys
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *
import random


import os
import torch.optim as optim
from models.resnet import ResNet18
from models.wideresnet import WideResNet
from models.preact_resnet import PreActResNet18
from collections import namedtuple
from utils_awp import AdvWeightPerturb
from torch.distributions import Categorical, Bernoulli
import copy
import os
import numpy
import argparse
import logging

cifar100_mean = (0.0, 0.0, 0.0)
cifar100_std = (1.0, 1.0, 1.0)
mu = torch.tensor(cifar100_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar100_std).view(3, 1, 1).cuda()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def normalize(X):

    return (X - mu)/std


upper_limit, lower_limit = 1,0


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
        return ({'input': x.cuda().float(), 'target': y.cuda().long()} for (x,y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm= "l_inf", early_stop=False,
               mixup=False, y_a=None, y_b=None, lam=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(normalize(X + delta))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if mixup:
                criterion = nn.CrossEntropyLoss()
                loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            all_loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
        else:
            all_loss = F.cross_entropy(model(normalize(X+delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='WideResNet')
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--l1', default=0, type=float)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--batch-size-test', default=128, type=int)
    parser.add_argument('--data-dir', default='../../Datasets/CIFAR100', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr-schedule', default='piecewise', choices=['superconverge', 'piecewise', 'linear', 'piecewisesmoothed', 'piecewisezoom', 'onedrop', 'multipledecay', 'cosine', 'cyclic'])
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--lr-one-drop', default=0.01, type=float)
    parser.add_argument('--lr-drop-epoch', default=100, type=int)
    parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'fgsm', 'free', 'none'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--attack-iters-test', default=20, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=float)
    parser.add_argument('--fgsm-alpha', default=1.25, type=float)
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fgsm-init', default='random', choices=['zero', 'random', 'previous'])
    parser.add_argument('--out-dir', default='./AWP/cifar100', type=str)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--width-factor', default=10, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--cutout', action='store_true')
    parser.add_argument('--cutout-len', type=int)
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--mixup-alpha', type=float)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--chkpt-iters', default=10, type=int)
    parser.add_argument('--awp-gamma', default=0.01, type=float)
    parser.add_argument('--awp-warmup', default=0, type=int)

    ## search
    parser.add_argument('--epsilon_types', type=list, default=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) # Same range as LAS-AT
    parser.add_argument('--epsilon_mode', type=str, default='entropy', help="choice for how to change epsilon, [fixed, linear, cyclic]")
    parser.add_argument('--rho_high', type=float, default=4.582, help= "threshold of the entropy")
    parser.add_argument('--rho_low',  type=float, default=3.35) 

    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--factor', default=0.7, type=float, help='Label Smoothing')
    parser.add_argument('--clip_grad_norm', default=1.0, type=float)

    return parser.parse_args()

eps = np.finfo(np.float32).eps.item()
start_epsilon = (8.0 / 255.)
start_alpha = (2.0 / 255.)
def _label_smoothing(label, factor):
    one_hot = np.eye(100)[label.cuda().data.cpu().numpy()]

    result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(100 - 1))

    return result


def LabelSmoothLoss(input, target):
    log_prob = F.log_softmax(input, dim=-1)
    loss = (-target * log_prob).sum(dim=-1).mean()
    return loss



def attack_batch(model, X, y, epsilon, alpha, iters):
    criterion = nn.CrossEntropyLoss()
    X, y = X.cuda(), y.cuda()
    delta = torch.zeros_like(X).cuda()
    delta.uniform_(-start_epsilon, start_epsilon)
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    for _ in range(iters):
        output = model(normalize(X + delta))
        loss = criterion(output, y)
        loss.backward()
        grad = delta.grad.detach()
        delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.grad.zero_()
    delta = delta.detach()
    return delta

def compute_entropy(x):
    """
    x is the probability distribution has the shape of [n, c]
    n is the batch size and c is the number of outputs(num of classes)
    """
    n = x.shape[0]
    h = torch.zeros(n).cuda()
    for i in range(n):
        for p in x[i]:
            h[i] += p * torch.log(p)
    return -h


def cyclical_epsilon(args, epoch, step_size=1):
    """
    The step size means after how many epochs the value of epsilon will change
    """
    epsilon_list = sorted(args.epsilon_types)
    min_eps = epsilon_list[0]
    max_eps = epsilon_list[-1]
    length = len(epsilon_list)
    idx = int((epoch/step_size) % length)
    return epsilon_list[idx]

def Get_delta(args, epoch, entropy_batch, input_batch, y_batch, target_model):
    # Use the stronger attack in the last 5 epochs

    batch_size = input_batch.shape[0]
    big_epsilon = [item for item in args.epsilon_types if item >12]
    epsilon = cyclical_epsilon(args, epoch, 1)
        
    target_model.eval()
    inputs, targets = input_batch.cuda(), y_batch.cuda()
    epsilon_batch = torch.ones(batch_size, 3, 1, 1).cuda()
    epsilon = (epsilon/255.) / std
    epsilon_batch = epsilon_batch * epsilon
    # use the smallest epsilon for the examples with large entropy
    epsilon_batch[entropy_batch > args.rho_high] = (sorted(args.epsilon_types)[0]/255. /std)
    epsilon_batch[entropy_batch < args.rho_low] = (random.choice(big_epsilon)/255. /std)
    

    iters = 10
    alpha = epsilon/iters * 2
    delta = attack_batch(target_model, inputs, targets, epsilon_batch, alpha, iters)
    return inputs + delta





def main():
    args = get_args()
    if args.awp_gamma <= 0.0:
        args.awp_warmup = np.infty

    out_dir = os.path.join(args.out_dir, args.epsilon_mode)
    out_dir = os.path.join(out_dir,'model_' + args.model)
    #
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(out_dir, 'eval.log' if args.eval else 'output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    """ 
    transforms = [Crop(32, 32), FlipLR()]
    if args.cutout:
        transforms.append(Cutout(args.cutout_len, args.cutout_len))
    if args.val:
        try:
            dataset = torch.load("cifar10_validation_split.pth")
        except:
            print("Couldn't find a dataset with a validation split, did you run "
                  "generate_validation.py?")
            return
        val_set = list(zip(transpose(dataset['val']['data']/255.), dataset['val']['labels']))
        val_batches = Batches(val_set, args.batch_size, shuffle=False, num_workers=2)
    else:
        dataset = cifar10(args.data_dir)
    """
    epsilon = (args.epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)

    if args.model == 'PreActResNet18':
        model = PreActResNet18()
        proxy = PreActResNet18()
    
    elif args.model == "ResNet18":
        model = ResNet18(num_classes=100)
        proxy = ResNet18(num_classes=100)
    elif args.model == 'WideResNet':
        model = WideResNet(num_classes=100)
        proxy = WideResNet(num_classes=100)
    else:
        raise ValueError("Unknown model")

    model = nn.DataParallel(model).cuda()
    proxy = nn.DataParallel(proxy).cuda()


    if args.l2:
        decay, no_decay = [], []
        for name,param in model.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                decay.append(param)
            else:
                no_decay.append(param)
        params = [{'params':decay, 'weight_decay':args.l2},
                  {'params':no_decay, 'weight_decay': 0 }]
    else:
        params = model.parameters()

    opt = torch.optim.SGD(params, lr=args.lr_max, momentum=0.9, weight_decay=5e-4)

    proxy_opt = torch.optim.SGD(proxy.parameters(), lr=0.01)
    awp_adversary = AdvWeightPerturb(model=model, proxy=proxy, proxy_optim=proxy_opt, gamma=args.awp_gamma)

    criterion = nn.CrossEntropyLoss()

    if args.attack == 'free':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
        delta.requires_grad = True
    elif args.attack == 'fgsm' and args.fgsm_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
        delta.requires_grad = True

    if args.attack == 'free':
        epochs = int(math.ceil(args.epochs / args.attack_iters))
    else:
        epochs = args.epochs


    if args.lr_schedule == 'superconverge':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
    elif args.lr_schedule == 'piecewise':
        def lr_schedule(t):
            if t / args.epochs < 0.5:
                return args.lr_max
            elif t / args.epochs < 0.75:
                return args.lr_max / 10.
            else:
                return args.lr_max / 100.

    elif args.lr_schedule == 'linear':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs], [args.lr_max, args.lr_max, args.lr_max / 10, args.lr_max / 100])[0]
    elif args.lr_schedule == 'onedrop':
        def lr_schedule(t):
            if t < args.lr_drop_epoch:
                return args.lr_max
            else:
                return args.lr_one_drop
    elif args.lr_schedule == 'multipledecay':
        def lr_schedule(t):
            return args.lr_max - (t//(args.epochs//10))*(args.lr_max/10)
    elif args.lr_schedule == 'cosine':
        def lr_schedule(t):
            return args.lr_max * 0.5 * (1 + np.cos(t / args.epochs * np.pi))
    elif args.lr_schedule == 'cyclic':
        lr_schedule = lambda t: np.interp([t], [0, 0.4 * args.epochs, args.epochs], [0, args.lr_max, 0])[0]


    best_test_robust_acc = 0
    best_val_robust_acc = 0
    train_loader, test_loader = get_loaders_cifar100(args.data_dir, args.batch_size)

    if args.resume:
        start_epoch = args.resume
        model.load_state_dict(torch.load(os.path.join(out_dir, f'model_{start_epoch-1}.pth')))
        opt.load_state_dict(torch.load(os.path.join(out_dir, f'opt_{start_epoch-1}.pth')))
        logger.info(f'Resuming at epoch {start_epoch}')

        if os.path.exists(os.path.join(out_dir, f'model_best.pth')):
            best_test_robust_acc = torch.load(os.path.join(out_dir, f'model_best.pth'))['test_robust_acc']
        if args.val:
            best_val_robust_acc = torch.load(os.path.join(out_dir, f'model_val.pth'))['val_robust_acc']
    else:
        start_epoch = 0

    if args.eval:
        if not args.resume:
            logger.info("No model loaded to evaluate, specify with --resume FNAME")
            return
        logger.info("[Evaluation mode]")

    logger.info('Epoch \t Train Time \t Test Time \t LR \t \t Train Loss \t Train Acc \t Train Robust Loss \t Train Robust Acc \t Test Loss \t Test Acc \t Test Robust Loss \t Test Robust Acc')
    
    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_robust_loss = 0
        train_robust_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()        
            lr = lr_schedule(epoch + (i + 1) / len(train_loader))
            
            opt.param_groups[0].update(lr=lr)

            with torch.no_grad():
                logits = model(X)
                prob = F.softmax(logits, dim=1)
                entropy_batch = compute_entropy(prob)

            if args.attack == 'pgd':
                # Random initialization
                if args.mixup:
                    temp=0
                else:
                    adv_examples = Get_delta(args, epoch, entropy_batch, X, y, model)
                    pocliy_inputs1 = adv_examples
                    delta = torch.clamp(pocliy_inputs1-X, min=-epsilon, max=epsilon)
                delta = delta.detach()
            elif args.attack == 'fgsm':
                delta = attack_pgd(model, X, y, epsilon, args.fgsm_alpha*epsilon, 1, 1, args.norm)
            # Standard training
            elif args.attack == 'none':
                delta = torch.zeros_like(X)
            X_adv = normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit))

            model.train()
            # calculate adversarial weight perturbation and perturb it
            if epoch >= args.awp_warmup:
                # not compatible to mixup currently.
                assert (not args.mixup)
                awp = awp_adversary.calc_awp(inputs_adv=X_adv,
                                             targets=y)
                awp_adversary.perturb(awp)

            robust_output = model(X_adv)
            robust_loss = criterion(robust_output, y)

            if args.l1:
                for name,param in model.named_parameters():
                    if 'bn' not in name and 'bias' not in name:
                        robust_loss += args.l1*param.abs().sum()

            opt.zero_grad()
            robust_loss.backward()
            opt.step()

            if epoch >= args.awp_warmup:
                awp_adversary.restore(awp)

            output = model(normalize(X))

            loss = criterion(output, y)

            train_robust_loss += robust_loss.item() * y.size(0)
            train_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

        train_time = time.time()

        model.eval()
        test_loss = 0
        test_acc = 0
        test_robust_loss = 0
        test_robust_acc = 0
        test_n = 0
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()

            # Random initialization
            if args.attack == 'none':
                delta = torch.zeros_like(X)
            else:
                delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters_test, args.restarts, args.norm, early_stop=args.eval)
            delta = delta.detach()

            robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
            robust_loss = criterion(robust_output, y)

            output = model(normalize(X))
            loss = criterion(output, y)

            test_robust_loss += robust_loss.item() * y.size(0)
            test_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            test_n += y.size(0)

        test_time = time.time()


        if not args.eval:
            logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f',
                epoch, train_time - start_time, test_time - train_time, lr,
                train_loss/train_n, train_acc/train_n, train_robust_loss/train_n, train_robust_acc/train_n,
                test_loss/test_n, test_acc/test_n, test_robust_loss/test_n, test_robust_acc/test_n)


            # save checkpoint
            if (epoch+1) % args.chkpt_iters == 0 or epoch+1 == epochs:
                torch.save(model.state_dict(), os.path.join(out_dir,'model_.pth'))
                torch.save(opt.state_dict(), os.path.join(out_dir, 'opt.pth'))

            # save best
            if test_robust_acc/test_n > best_test_robust_acc:
                torch.save({
                        'state_dict':model.state_dict(),
                        'test_robust_acc':test_robust_acc/test_n,
                        'test_robust_loss':test_robust_loss/test_n,
                        'test_loss':test_loss/test_n,
                        'test_acc':test_acc/test_n,
                    }, os.path.join(out_dir, f'model_best.pth'))
                best_test_robust_acc = test_robust_acc/test_n
        else:
            logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f',
                epoch, train_time - start_time, test_time - train_time, -1,
                -1, -1, -1, -1,
                test_loss/test_n, test_acc/test_n, test_robust_loss/test_n, test_robust_acc/test_n)
            return


if __name__ == "__main__":
    main()
