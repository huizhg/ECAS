import torch
from autoattack import AutoAttack
from models.resnet import ResNet18
from models.wideresnet import WideResNet
from utils.utils import *
import argparse
import sys
import os
import logging
#sys.path.insert(0, '..')
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../Dataset/CIFAR100')
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--epsilon', type=int, default=8)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--model', default='WideResNet', type=str, help='model name')
    parser.add_argument('--n_ex', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--out_dir', type=str, default='./test_out_cifar100')
    parser.add_argument('--method_name', type=str, default='pgd-entropy')

    arguments = parser.parse_args()
    return arguments


args = get_args()
out_dir = os.path.join(args.out_dir, args.method_name)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
logfile1 = os.path.join(out_dir, 'log_file1.txt')
logfile2 = os.path.join(out_dir, 'log_file2.txt')
if os.path.exists(logfile1):
    os.remove(logfile1)
if os.path.exists(logfile2):
    os.remove(logfile2)

accuracy_test_log = os.path.join(out_dir, 'output.log')
if os.path.exists(accuracy_test_log):
    os.remove(accuracy_test_log)

logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    filename=accuracy_test_log)
logger.info(args)






device = 'cuda' if torch.cuda.is_available() else 'cpu'


if args.model == "ResNet18":
    target_model = ResNet18(num_classes=100)
elif args.model == "WideResNet":
    target_model = WideResNet(num_classes=100)

target_model = target_model.to(device)
checkpoint = torch.load(args.model_path)
from collections import OrderedDict
try:
    # 'net' for models trained by PGD-AT and TRADES, 'state_dict' for models trained by AWP
    target_model.load_state_dict(checkpoint['net'])  
except:
    new_state_dict = OrderedDict()
    for k, v in checkpoint['net'].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    target_model.load_state_dict(new_state_dict, False)
#torch.save(target_model.state_dict(), '/apdcephfs/private_xiaojunjia/LAS-AT/weights/LAS-AT/CIFAR100/LAS_Trades/model.pth')


target_model.eval()



train_loader, test_loader = get_loaders_cifar100(args.data_dir, args.batch_size)
epsilon=args.epsilon
epsilon=float(epsilon)/255.
print(epsilon)


#AT_fgsm_loss,AT_fgsm_acc=evaluate_fgsm(test_loader, target_model, 1)
AT_pgd_loss_10, AT_pgd_acc_10 = evaluate_pgd(test_loader, target_model, 10, 1, epsilon / std)

print('AT_pgd_acc_10:', AT_pgd_acc_10)
logger.info(f'AT_pgd_acc_10: {AT_pgd_acc_10}')

AT_pgd_loss_20, AT_pgd_acc_20 = evaluate_pgd(test_loader, target_model, 20, 1, epsilon / std)
print('AT_pgd_acc_20:', AT_pgd_acc_20)
logger.info(f'AT_pgd_acc_20: {AT_pgd_acc_20}' )

AT_pgd_loss_50, AT_pgd_acc_50 = evaluate_pgd(test_loader, target_model, 50, 1, epsilon / std)
print('AT_pgd_acc_50:', AT_pgd_acc_50)
logger.info(f'AT_pgd_acc_50:{AT_pgd_acc_50}')


AT_CW_loss_20, AT_pgd_cw_acc_20 = evaluate_pgd_cw(test_loader, target_model, 20, 1)
print('AT_pgd_cw_acc_20:', AT_pgd_cw_acc_20)
logger.info(f'AT_pgd_cw_acc_20:{AT_pgd_cw_acc_20}')


AT_models_test_loss, AT_models_test_acc = evaluate_standard(test_loader, target_model)
print('AT_models_test_acc:', AT_models_test_acc)
#
logger.info(f'AT_models_test_acc:{ AT_models_test_acc}')
#print('AT_fgsm_acc:', AT_fgsm_acc)

adversary1 = AutoAttack(target_model, norm=args.norm, eps=epsilon, version='standard',log_path=logfile1)

#adversary2 = AutoAttack(target_model, norm=args.norm, eps=epsilon, version='standard',log_path=logfile2)
l = [x for (x, y) in test_loader]
x_test = torch.cat(l, 0)
l = [y for (x, y) in test_loader]
y_test = torch.cat(l, 0)

adv_complete = adversary1.run_standard_evaluation(x_test[:args.n_ex], y_test[:args.n_ex],
                bs=args.batch_size)
# adv_complete1 = adversary2.run_standard_evaluation_individual(x_test[:args.n_ex], y_test[:args.n_ex],
#                 bs=args.batch_size)