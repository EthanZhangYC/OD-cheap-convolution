

import os
import utils.common as utils
from importlib import import_module
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR

from utils.options import args
from model.cifar10.shiftresnet import *
import torch.backends.cudnn as cudnn


def _make_dir(path):
    if not os.path.exists(path): os.makedirs(path)

ckpt = utils.checkpoint(args)
print_logger = utils.get_logger(os.path.join(args.job_dir, "logger.log"))
utils.print_params(vars(args), print_logger.info)
writer_train = SummaryWriter(args.job_dir + '/run/train')
writer_test = SummaryWriter(args.job_dir + '/run/test')


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cudnn.benchmark = True

    start_epoch = args.start_epoch

    lr_decay_step = list(map(int, args.lr_decay_step.split(',')))

    # Data loading
    print_logger.info('=> Preparing data..')
    loader = import_module('data.' + args.dataset).Data(args)

    num_classes=0
    if args.dataset in ['cifar10']:
        num_classes = 10

    model = eval(args.block_type+'ResNet56_od')(groups=args.group_num, expansion=args.expansion,
                                     num_stu=args.num_stu, num_classes=num_classes).cuda()

    if len(args.gpu)>1:
        device_id=[]
        for i in range((len(args.gpu)+1)//2):
            device_id.append(i)
        model=torch.nn.DataParallel(model, device_ids=device_id)

    best_prec = 0.0

    if not model:
        print_logger.info("Model arch Error")
        return

    print_logger.info(model)

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=lr_decay_step, gamma=args.lr_decay_factor)

    # Optionally resume from a checkpoint
    resume = args.resume
    if resume:
        print('=> Loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume)
        state_dict = checkpoint['state_dict']
        if args.adjust_ckpt:
            new_state_dict={k.replace('module.', ''): v for k, v in state_dict.items()}
        else:
            new_state_dict=state_dict

        if args.start_epoch==0:
            start_epoch = checkpoint['epoch']

        best_prec = checkpoint['best_prec']
        model.load_state_dict(new_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print('=> Continue from epoch {}...'.format(start_epoch))

    if args.test_only:
        test_prec = test(args,loader.loader_test, model)
        print('=> Test Prec@1: {:.2f}'.format(test_prec[0]))
        return

    record_top5=0.
    for epoch in range(start_epoch, args.epochs):

        scheduler.step(epoch)

        train_loss, train_prec = train(args, loader.loader_train, model, criterion, optimizer, epoch)
        test_prec = test(args, loader.loader_test, model, epoch)

        is_best = best_prec < test_prec[0]
        if is_best:
            record_top5=test_prec[1]
        best_prec = max(test_prec[0], best_prec)

        state = {
                'state_dict': model.state_dict(),
                'test_prec': test_prec[0],
                'best_prec': best_prec,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1
            }

        if epoch % args.save_freq==0 or is_best:
            ckpt.save_model(state, epoch + 1, is_best)
        print_logger.info("=>Best accuracy {:.3f}, {:.3f}".format(best_prec, record_top5))

def train(args, loader_train, model, criterion, optimizer, epoch):

    losses = utils.AverageMeter()
    top1_t = utils.AverageMeter()
    top5_t = utils.AverageMeter()
    top1_s = utils.AverageMeter()
    top5_s = utils.AverageMeter()

    model.train()
    
    # update learning rate
    for param_group in optimizer.param_groups:
        writer_train.add_scalar(
            'learning_rate', param_group['lr'], epoch
        )

    num_iterations = len(loader_train)

    for i, (inputs, targets) in enumerate(loader_train, 1):

        inputs = inputs.cuda()
        targets = targets.cuda()

        # compute output
        logits_s, logits_t = model(inputs)

        loss = criterion(logits_t, targets)
        best_prec_s_1=torch.tensor(0.).cuda()
        best_prec_s_5=torch.tensor(0.).cuda()
        best_branch=1
        for j in range(args.num_stu):
            loss += criterion(logits_s[j], targets)
            loss += args.t * args.t * utils.KL(logits_t / args.t,logits_s[j] / args.t)
            prec1, prec5 = utils.accuracy(logits_s[j], targets, topk=(1, 5))
            writer_train.add_scalar(
                'train_stu_%d_top1'%(j+1), prec1.item(), num_iterations * epoch + i
            )
            if prec1>best_prec_s_1:
                best_prec_s_1=prec1
                best_prec_s_5=prec5
                best_branch=j+1

        prec1=best_prec_s_1
        prec5=best_prec_s_5

        prec1_t, prec5_t = utils.accuracy(logits_t, targets, topk=(1, 5))
        top1_t.update(prec1_t.item(), inputs.size(0))
        top5_t.update(prec5_t.item(), inputs.size(0))

        losses.update(loss.item(), inputs.size(0))

        writer_train.add_scalar(
            'train_top1', prec1.item(), num_iterations * epoch + i
            )
        writer_train.add_scalar(
            'train_loss', loss.item(), num_iterations * epoch + i
            )

        top1_s.update(prec1.item(), inputs.size(0))
        top5_s.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        if i % args.print_freq == 0:
            print_logger.info(
                'Epoch[{0}]({1}/{2}): '
                'Loss {loss.avg:.4f} '
                'TeacherPrec@1(1,5) {top1_t.avg:.2f}, {top5_t.avg:.2f} '
                'StuPrec@1(1,5) {top1_s.avg:.2f}, {top5_s.avg:.2f} '
                'Best branch: {best_branch: d}'.format(
                    epoch, i, num_iterations, loss=losses,
                    top1_t=top1_t, top5_t=top5_t,
                    top1_s=top1_s, top5_s=top5_s, best_branch=best_branch))

    return losses.avg, top1_s.avg

def test(args,loader_test, model, epoch=0):

    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    top1_t = utils.AverageMeter()
    top5_t = utils.AverageMeter()

    top1_s1 = utils.AverageMeter()
    top1_s2 = utils.AverageMeter()
    top1_s3 = utils.AverageMeter()
    top1_s4 = utils.AverageMeter()

    top5_s1 = utils.AverageMeter()
    top5_s2 = utils.AverageMeter()
    top5_s3 = utils.AverageMeter()
    top5_s4 = utils.AverageMeter()

    model.eval()
    num_iterations = len(loader_test)

    with torch.no_grad():
        print_logger.info("=> Evaluating...")

        for i, (inputs, targets) in enumerate(loader_test, 1):

            inputs = inputs.cuda()
            targets = targets.cuda()

            # compute output
            logits_s, logits_t = model(inputs)
            best_prec_s_1 = 0.
            for j in range(args.num_stu):
                prec1, prec5 = utils.accuracy(logits_s[j], targets, topk=(1, 5))
                eval('top1_s%d'%(j+1)).update(prec1[0], inputs.size(0))
                eval('top5_s%d' % (j + 1)).update(prec5[0], inputs.size(0))
                if prec1 > best_prec_s_1:
                    best_prec_s_1 = prec1

                writer_test.add_scalar(
                    'test_stu_%d_top1' % (j + 1), prec1[0], num_iterations * epoch + i
                )

            prec1, prec5 = utils.accuracy(logits_t, targets, topk=(1, 5))
            writer_test.add_scalar(
                'test_tea_top1', prec1[0], num_iterations * epoch + i
            )
            top1_t.update(prec1[0], inputs.size(0))
            top5_t.update(prec5[0], inputs.size(0))

        for j in range(args.num_stu):
            if eval('top1_s%d'%(j+1)).avg > top1.avg:
                top1.avg = eval('top1_s%d'%(j+1)).avg
                top5.avg = eval('top5_s%d'%(j+1)).avg
                #best_branch = j+ 1

        print_logger.info(
            'Epoch[{0}]({1}/{2}): '
            'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
            epoch, i, num_iterations, top1=top1, top5=top5))

        for i in range(args.num_stu):
            print_logger.info('top1_s%d: %.2f'%(i+1, eval('top1_s%d'%(i+1)).avg))
        print_logger.info('top1_t: %.2f' % (top1_t.avg))

    if not args.test_only:
        writer_test.add_scalar('test_top1', top1.avg, epoch)

    return top1.avg, top5.avg

if __name__ == '__main__':
    main()

        
        




