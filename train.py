import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from models.ST_Former import DECNet
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import datetime
from dataloader.dataset_PPB import DECNet_train_data_loader, DECNet_test_data_loader
from sklearn.metrics import f1_score, classification_report

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'    #使用的gpu

seed = 42
cudnn.benchmark = False  # if benchmark=True, deterministic will be False
cudnn.deterministic = True
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--data_set', type=int, default=1)
parser.add_argument('-t', default=3, type=int, help='cut 30s to t(s)')

parser.add_argument('--train_txt_path', default='PPB_CIR_V_DB_rgb_112_3s_train_fold_0.txt', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--test_txt_path', default='PPB_CIR_V_DB_rgb_112_3s_test_fold_0.txt', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--contrast', default='V-DB', type=str, help='若为true, 则DB全为0')
#超参数
parser.add_argument('--s_former_depth', default=1, type=int)
parser.add_argument('--t_former_depth', default=3, type=int)
parser.add_argument('--nf', default=32, type=int)
parser.add_argument('--Incep_depth', default=6, type=int)

args = parser.parse_args()

assert args.contrast in ['V-DB', 'V', 'DB']

now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H:%M:%S.%f]-")
log_txt_path = './log/' + time_str + 'PPB' + '-set' + str(args.data_set) + f'-{args.t}s' + '-log.txt'
log_curve_path = './log/' + time_str + 'PPB' + '-set' + str(args.data_set) + f'{-args.t}s' + '-log.png'
checkpoint_path = './checkpoint/' + "PPB" + time_str + 'set' + str(args.data_set) + f'-{args.t}s' + '-model.pth'
best_checkpoint_path = './checkpoint/' + "PPB" + time_str + 'set' + str(args.data_set) + f'-{args.t}s' + '-model_best.pth'

train_txt_path = args.train_txt_path
test_txt_path = args.test_txt_path


print('contrast:', args.contrast)
print('train_txt_path:', train_txt_path)
print('test_txt_path:', test_txt_path)
with open(log_txt_path, 'a') as f:
    f.write('contrast: ' + str(args.contrast) + '\n')
    f.write('train_txt_path: ' + train_txt_path + '\n')
    f.write('test_txt_path: ' + test_txt_path + '\n')
    f.write('args:' + str(args) + '\n')         
# args.epochs = args.epochs * args.t

def main():
    best_acc = 0
    recorder = RecorderMeter(args.epochs)
    print('The training time: ' + now.strftime("%m-%d %H:%M"))
    print('The training set: set ' + str(args.data_set))
    with open(log_txt_path, 'a') as f:
        f.write('The training set: set ' + str(args.data_set) + '\n')

    # create model and load pre_trained parameters
    model = DECNet(s_former_depth=args.s_former_depth, t_former_depth=args.t_former_depth, nf=args.nf, Incep_depth=args.Incep_depth)
    model = torch.nn.DataParallel(model).cuda()

    with open(log_txt_path, 'a') as f:
        f.write('model:' + str(model) + '\n')

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40 * args.t, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            # args.start_epoch = checkpoint['epoch']
            # best_acc = checkpoint['best_acc']
            # recorder = checkpoint['recorder']
            # best_acc = best_acc.cuda()
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    # cudnn.benchmark = True

    # Data loading code
    train_data = DECNet_train_data_loader(data_set=args.data_set, txt_path=train_txt_path, contrast=args.contrast, t=args.t)
    test_data = DECNet_test_data_loader(data_set=args.data_set, txt_path=test_txt_path, contrast=args.contrast, t=args.t)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=False)
    val_loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True,
                                             drop_last=False)

    for epoch in range(args.start_epoch, args.epochs):
        inf = '********************' + str(epoch) + '********************'
        start_time = time.time()
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']

        with open(log_txt_path, 'a') as f:
            f.write(inf + '\n')
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n')

        print(inf)
        print('Current learning rate: ', current_learning_rate)

        # train for one epoch
        train_acc, train_los = train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        val_acc, val_los = validate(val_loader, model, criterion, args)

        scheduler.step()

        # remember best acc and save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'recorder': recorder}, is_best)

        # print and save log
        epoch_time = time.time() - start_time
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        recorder.plot_curve(log_curve_path)

        print('The best accuracy: {:.3f}'.format(best_acc.item()))
        print('An epoch time: {:.1f}s'.format(epoch_time))
        with open(log_txt_path, 'a') as f:
            f.write('The best accuracy: ' + str(best_acc.item()) + '\n')
            f.write('An epoch time: {:.1f}s' + str(epoch_time) + '\n')                                                                                                                                                                                                                                                                            


def train(train_loader, model, criterion, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    loss_v_epoch, loss_db_epoch, loss_final_epoch, loss_total_epoch, total_acc_train_v, total_acc_train_db= 0, 0, 0, 0, 0, 0
    for i, (DB_feature, (images, target)) in enumerate(train_loader):

        DB_feature = DB_feature.cuda()
        images = images.cuda()
        target = target.cuda()

        
        output_v, output_db, output = model((images, DB_feature))
        
        loss_v = criterion(output_v, target)
        loss_db = criterion(output_db, target)
        loss_final = criterion(output, target)
        loss = loss_v + loss_db + loss_final

        loss_v_epoch += loss_v.item()
        loss_db_epoch += loss_db.item()
        loss_final_epoch += loss_final.item()
        loss_total_epoch += loss.item()

        accuracy_train_v = (output_v.argmax(1) == target).sum()
        accuracy_train_db = (output_db.argmax(1) == target).sum()
        total_acc_train_v += accuracy_train_v
        total_acc_train_db += accuracy_train_db


        # measure accuracy and record loss
        acc1, _ = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i)
    acc_train_v = total_acc_train_v / len(train_loader.dataset)
    acc_train_db = total_acc_train_db / len(train_loader.dataset)

    print('train: loss_v: {:.4f}, loss_db: {:.4f}, loss_final: {:.4f}, loss_total: {:.4f}, acc_train_v: {:.4f}, acc_train_db: {:.4f}'.format(loss_v_epoch, loss_db_epoch, loss_final_epoch, loss_total_epoch, acc_train_v, acc_train_db))
    with open(log_txt_path, 'a') as f:
            f.write('train: loss_v: {:.4f}, loss_db: {:.4f}, loss_final: {:.4f}, loss_total: {:.4f}, acc_train_v: {:.4f}, acc_train_db: {:.4f}'.format(loss_v_epoch, loss_db_epoch, loss_final_epoch, loss_total_epoch, acc_train_v, acc_train_db) + '\n')
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    loss_v_epoch, loss_db_epoch, loss_final_epoch, loss_total_epoch, total_acc_test_v, total_acc_test_db = 0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for i, (DB_feature, (images, target)) in enumerate(val_loader):
            DB_feature = DB_feature.cuda()
            images = images.cuda()
            target = target.cuda()

            output_v, output_db, output = model((images, DB_feature))
            loss_v = criterion(output_v, target)
            loss_db = criterion(output_db, target)
            loss_final = criterion(output, target)
            loss = loss_v + loss_db + loss_final

            loss_v_epoch += loss_v.item()
            loss_db_epoch += loss_db.item()
            loss_final_epoch += loss_final.item()
            loss_total_epoch += loss.item()
            
            accuracy_test_v = (output_v.argmax(1) == target).sum()
            accuracy_test_db = (output_db.argmax(1) == target).sum()
            total_acc_test_v += accuracy_test_v
            total_acc_test_db += accuracy_test_db


            _, preds = torch.max(output, dim=1)
            # 将预测结果和真实标签添加到列表中
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            # measure accuracy and record loss
            acc1, _ = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            if i % args.print_freq == 0:
                progress.display(i)
        acc_test_v = total_acc_test_v / len(val_loader.dataset)
        acc_test_db = total_acc_test_db / len(val_loader.dataset)  
        print('val: loss_v: {:.4f}, loss_db: {:.4f}, loss_final: {:.4f}, loss_total: {:.4f}, acc_test_v: {:.4f}, acc_test_db: {:.4f}'.format(loss_v_epoch, loss_db_epoch, loss_final_epoch, loss_total_epoch, acc_test_v, acc_test_db))
        # TODO: this should also be done with the ProgressMeter
        print('Current Accuracy: {top1.avg:.3f}'.format(top1=top1))
        with open(log_txt_path, 'a') as f:
            f.write('Current Accuracy: {top1.avg:.3f}'.format(top1=top1) + '\n')
            f.write('val: loss_v: {:.4f}, loss_db: {:.4f}, loss_final: {:.4f}, loss_total: {:.4f}, acc_test_v: {:.4f}, acc_test_db: {:.4f}'.format(loss_v_epoch, loss_db_epoch, loss_final_epoch, loss_total_epoch, acc_test_v, acc_test_db) + '\n')
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)    
    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    micro_f1 = f1_score(all_targets, all_preds, average='micro')
    weighted_f1 = f1_score(all_targets, all_preds, average='weighted')
    class_7_report = classification_report(all_targets, all_preds, zero_division=0, digits=4)
    print('macro_f1: {:.4f}, micro_f1: {:.4f}, weighted_f1: {:.4f}'.format(macro_f1, micro_f1, weighted_f1))
    print('class_7_report: \n', class_7_report)
    
    # 计算每个类别的准确率
    class_correct = list(0. for i in range(7))
    class_total = list(0. for i in range(7))
    for i in range(len(all_preds)):
        label = all_targets[i]
        class_correct[label] += int(all_preds[i] == all_targets[i])
        class_total[label] += 1
    for i in range(7):
        print('Accuracy of %5s : %.2f %%' % (i, 100 * class_correct[i] / class_total[i]))
    
    with open(log_txt_path, 'a') as f:
        f.write('macro_f1: {:.4f}, micro_f1: {:.4f}, weighted_f1: {:.4f}'.format(macro_f1, micro_f1, weighted_f1) + '\n')
        f.write('class_7_report: \n' + str(class_7_report) + '\n')
        for i in range(7):
            f.write('Accuracy of %5s : %.2f %%' % (i, 100 * class_correct[i] / class_total[i]) + '\n')

    return top1.avg, losses.avg



def save_checkpoint(state, is_best):
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_checkpoint_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        with open(log_txt_path, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 50
        self.epoch_losses[idx, 1] = val_loss * 50
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1600, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 1
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            # print('Curve was saved')
        plt.close(fig)


if __name__ == '__main__':
    main()
