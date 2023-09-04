# encoding: utf-8
import argparse
from utils import *

from models.resnet import ResNet18
from models.wideresnet import WideResNet
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli
import copy
import os
import numpy
import argparse
import logging
import random
from torch.nn.utils import clip_grad_norm_
logger = logging.getLogger(__name__)
CUDA_LAUNCH_BLOCKING=1
#from tensorboardX import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser('Trades cifar10')
    # target model
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../../Datasets/CIFAR100', type=str)
    parser.add_argument('--out-dir', default='./TRADES/cifar100', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--target_model_lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--target_model_lr_scheduler', default='multistep', type=str, choices=['cyclic', 'multistep'])
    parser.add_argument('--target_model_lr_min', default=0., type=float)
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--model', default='WideResNet', type=str, help='model name')
    parser.add_argument('--weight-decay', '--wd', default=5e-4,
                        type=float, metavar='W')


    parser.add_argument('--path', default='./TRADES', type=str, help='model name')
    parser.add_argument('--epsilon_mode', type=str, default='entropy', help="choice for how to change epsilon, [fixed, linear, cyclic]")
    ## search
    parser.add_argument('--epsilon_types', type=list, default=[5,6,7,8,9])
    parser.add_argument('--rho_high', type=float, default=4.582, help= "threshold of the entropy")
    parser.add_argument('--rho_low',  type=float, default=3.35) 
    ## policy Hyperparameters
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    

    print('**')

    parser.add_argument('--interval_num', type=int, default=1)
    parser.add_argument('--exp_iter', type=int, default=1)

    parser.add_argument('--clip_grad_norm', default=1.0, type=float)

    arguments = parser.parse_args()
    return arguments


args = get_args()


out_dir=os.path.join(args.out_dir, args.epsilon_mode )
out_dir=os.path.join(out_dir,'model_'+args.model)




# writer = SummaryWriter(tensor_path)

eps = np.finfo(np.float32).eps.item()
criterion_kl = nn.KLDivLoss(size_average=False)




print(out_dir)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
logfile = os.path.join(out_dir, 'output.log')
if os.path.exists(logfile):
    os.remove(logfile)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    filename=logfile)
logger.info(args)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_loader, test_loader = get_loaders_cifar100(args.data_dir, args.batch_size)
best_acc = 0
best_clean_acc=0
start_epoch = 0


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X + delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta





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

def attack_batch(model, X, y, epsilon, alpha, iters):
    delta = torch.zeros_like(X).cuda()
    criterion_kl = nn.KLDivLoss(size_average=False)
    for i in range(epsilon.shape[0]):
        for j in range(epsilon.shape[1]):
            delta[i, j, :, :].uniform_(-epsilon[i][j][0][0].item(), epsilon[i][j][0][0].item())

    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
    criterion = nn.CrossEntropyLoss()
    delta.requires_grad = True
    x_adv = X.detach() + 0.001 * torch.randn(X.shape).cuda().detach()
    for _ in range(iters):
        x_adv.requires_grad_()
        with torch.enable_grad():
            adv_kl = F.log_softmax(model(x_adv), dim=1)
            clean_kl = F.softmax(model(X), dim=1)
            loss_kl = criterion_kl(adv_kl, clean_kl)

        grad = torch.autograd.grad(loss_kl,[x_adv])[0]
        x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, X - epsilon), X + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)


    delta = x_adv - X
    delta = delta.detach()
    return delta

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
    big_epsilon = [item for item in args.epsilon_types if item >7]
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
    beta = 6.0
    alpha = epsilon/iters * 2
    delta = attack_batch(target_model, inputs, targets, epsilon_batch, alpha, iters)
    target_model.train()
    return inputs + delta, beta



device = 'cuda' if torch.cuda.is_available() else 'cpu'


if args.model == "ResNet18":
    target_model = ResNet18(num_classes=100)

elif args.model == "WideResNet":
    target_model = WideResNet(num_classes=100)

device_id=range(torch.cuda.device_count())
if len(device_id)>1:
    target_model=torch.nn.DataParallel(target_model)
target_model = target_model.to(device)
criterion = nn.CrossEntropyLoss()
target_model_optimizer = optim.SGD(target_model.parameters(), lr=args.target_model_lr, momentum=0.9, weight_decay=args.weight_decay)
lr_steps = args.epochs * len(train_loader)

if args.target_model_lr_scheduler == 'cyclic':
    target_model_scheduler = torch.optim.lr_scheduler.CyclicLR(target_model_optimizer, base_lr=args.target_model_lr_min,
                                                               max_lr=args.target_model_lr,
                                                               step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
elif args.target_model_lr_scheduler == 'multistep':
    target_model_scheduler = torch.optim.lr_scheduler.MultiStepLR(target_model_optimizer,
                                                                  milestones=[int(lr_steps* 99/110), int(lr_steps * 104/ 100)],
                                                                  gamma=0.1)

target_model_path = os.path.join(out_dir, 'target_model_ckpt.t7')
from collections import OrderedDict
if os.path.exists(target_model_path):
        print("resuming............................................")
        logger.info("resuming............................................")
        #start_epoch = args.resume
        target_model_path = os.path.join(out_dir, 'target_model_ckpt.t7')
        target_model_checkpoint = torch.load(target_model_path)
        start_epoch = target_model_checkpoint['epoch']
        try:
            target_model.load_state_dict(target_model_checkpoint['net'])
        except:
            new_state_dict = OrderedDict()
            for k, v in target_model_checkpoint['net'].items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            target_model.load_state_dict(new_state_dict,False)

        target_model_optimizer_path=os.path.join(out_dir, 'target_model_optimizer.pth')
        target_model_optimizer.load_state_dict(torch.load(target_model_optimizer_path))
        #torch.save(policy_optimizer.state_dict(), os.path.join(out_dir, 'policy_model_optimizer.pth'))

        target_model_scheduler_path = os.path.join(out_dir, 'target_model_scheduler.pth')
        target_model_scheduler.load_state_dict(torch.load(target_model_scheduler_path))

        # if os.path.exists(os.path.join(args.fname, f'model_best.pth')):
        #     best_test_robust_acc = torch.load(os.path.join(args.fname, f'model_best.pth'))['test_robust_acc']
        # if args.val:
        #     best_val_robust_acc = torch.load(os.path.join(args.fname, f'model_val.pth'))['val_robust_acc']
        best_target_model_path = os.path.join(out_dir, 'best_target_model_ckpt.t7')
        best_target_model_checkpoint = torch.load(best_target_model_path)

        best_acc=best_target_model_checkpoint['best_acc']
        best_clean_acc = best_target_model_checkpoint['best_clean_acc']
        logger.info('Test Acc  \t PGD Acc')
        logger.info('%.4f \t  \t %.4f', best_clean_acc, best_acc)
else:
        start_epoch = 0


global curr_step
curr_step = 0

import time

train_acc_list = []
train_loss_list = []

def train(epoch):
    print('\nEpoch: %d' % epoch)
    # logger.info('\nEpoch: %d' % epoch)
    start_epoch_time = time.time()
    train_loss = 0
    train_acc = 0
    train_n = 0
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)


        global curr_step
        curr_step = curr_step + 1
    
        #####训练target model
        print("*******************train target model**********************")
        criterion_kl_not_sum = nn.KLDivLoss(size_average=False, reduce=False)

        pocliy_inputs1=inputs.clone().cuda()

        with torch.no_grad():
            logits = target_model(inputs)
            prob = F.softmax(logits, dim=1)
            entropy_batch = compute_entropy(prob)
        
        adv_examples,beta= Get_delta(args, epoch, entropy_batch, pocliy_inputs1, targets, target_model)
        pocliy_inputs1=adv_examples

        target_model.train()
        batch_size=len(inputs)
        logits = target_model(inputs)
        logits_adv=target_model(pocliy_inputs1)
        loss_natural = F.cross_entropy(logits, targets)
        all_loss_robust = beta * torch.sum(
            criterion_kl_not_sum(F.log_softmax(logits_adv, dim=1),
                                 F.softmax(logits, dim=1)), dim=1)
        #print(all_loss_robust.shape)
        loss_robust = (1.0 / batch_size) * torch.sum(all_loss_robust, dim=0)


        target_loss = loss_natural +loss_robust
        #train_acc = (logits.max(1)[1] == targets).sum().item()
        target_model_optimizer.zero_grad()
        target_loss.backward()
        target_model_optimizer.step()
        target_model_scheduler.step()
        train_loss += target_loss.item() * targets.size(0)
        train_acc += (logits_adv.max(1)[1] == targets).sum().item()
        train_n += targets.size(0)
        print("Target model loss:", target_loss)
    epoch_time = time.time()
    train_acc_list.append(train_acc/train_n)
    train_loss_list.append(train_loss/train_n)

    lr = target_model_scheduler.get_lr()[0]
    logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
                epoch, epoch_time - start_epoch_time, lr, train_loss / train_n, train_acc / train_n)


test_robust_acc_list=[]
test_clean_acc_list=[]
test_robust_loss_list=[]
test_clean_loss_list=[]


def test(epoch):
    global best_acc
    global best_clean_acc
    target_model.eval()
    test_loss = 0
    correct = 0
    total = 0
    pgd_loss, pgd_acc = evaluate_pgd(test_loader, target_model, 10, 1)
    test_loss, test_acc = evaluate_standard(test_loader, target_model)

    test_robust_acc_list.append(pgd_acc)
    test_clean_acc_list.append(test_acc)
    test_robust_loss_list.append(pgd_loss)
    test_clean_loss_list.append(test_loss)
    acc = pgd_acc
    state = {
        'net': target_model.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }


    target_path = os.path.join(out_dir, 'target_model_ckpt.t7')
    torch.save(state, target_path)
    torch.save(target_model_optimizer.state_dict(), os.path.join(out_dir, 'target_model_optimizer.pth'))
    torch.save(target_model_scheduler.state_dict(), os.path.join(out_dir, 'target_model_scheduler.pth'))
    # Save checkpoint.
    # Save checkpoint.

    # print('Test acc:', test_acc)
    # print('Val acc:', acc)
    # logger.info('Test acc: ', test_acc)
    # logger.info('Val acc: ', acc)
    if acc >=best_acc:

        print('Saving..')
        # logger.info("Saving..")
        state = {
            'net': target_model.state_dict(),
            'best_clean_acc':test_acc,
            'best_acc': acc,
            'epoch': epoch,
        }



        target_path = os.path.join(out_dir, 'best_target_model_ckpt.t7')
        torch.save(state, target_path)
        best_acc = acc
        best_clean_acc = test_acc

    print(best_acc)
    # logger.info(best_acc)
    # logger.info(test_acc)
    logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
    logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)
    logger.info('Test Acc  \t PGD Acc')
    logger.info('%.4f \t  \t %.4f',  best_clean_acc, best_acc)

    return best_acc


for epoch in range(start_epoch,  args.epochs):
    train(epoch)
    print("*****************************************test*************************")
    #logger.info(("*****************************************test*************************"))
    result_acc = test(epoch)
    print(result_acc)
    #logger.info(result_acc)
logger.info(test_robust_acc_list)
logger.info(test_clean_acc_list)

test_file_loss = os.path.join(out_dir, 'test_loss_stats.npy')
test_file_acc = os.path.join(out_dir, "test_acc_stats.npy")
np.save(test_file_loss, np.stack((np.array(test_clean_loss_list), np.array(test_robust_loss_list))))
np.save(test_file_acc, np.stack((np.array(test_clean_acc_list), np.array(test_robust_acc_list))))

train_loss = os.path.join(out_dir, "train_loss_stats.npy")
train_acc = os.path.join(out_dir, "train_acc_stats.npy")
np.save(train_loss, np.array(train_loss_list))
np.save(train_acc, np.array(train_acc_list))


print(test_robust_acc_list)
print(test_clean_acc_list)