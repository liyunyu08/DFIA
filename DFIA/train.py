import os
import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from common.meter import Meter
from common.utils import detect_grad_nan, compute_accuracy, set_seed, setup_run, get_resume_file
from models.dataloader.samplers import CategoriesSampler
from models.dataloader.data_utils import dataset_builder
from models.base_model import DFIA
from test import test_main, evaluate




def train(epoch, model, loader, optimizer, args=None):
    model.train()
    train_loader = loader['train_loader']
    train_loader_aux = loader['train_loader_aux']
    label = torch.arange(args.way).repeat(args.query).cuda()  # 012340123401234...
    label = label.type(torch.LongTensor)
    label = label.cuda()
    loss_meter = Meter()
    acc_meter = Meter()
    k = args.way * args.shot
    tqdm_gen = tqdm.tqdm(train_loader)


    for i, ((data, train_labels), (data_aux, train_labels_aux)) in enumerate(zip(tqdm_gen, train_loader_aux), 1):
        data, train_labels = data.cuda(), train_labels.cuda()
        data_aux, train_labels_aux = data_aux.cuda(), train_labels_aux.cuda()


        model.module.mode = 'encoder'
        freq_fuse,spa = model(data,Re=True)
        _,data_aux = model(data_aux,Re=True)
        freq_fuse_shot, freq_fuse_query = freq_fuse[:k], freq_fuse[k:]
        data_shot, data_query = spa[:k], spa[k:]

        if args.shot > 1:
            freq_fuse_shot = freq_fuse_shot.contiguous().view(args.shot, args.way, *freq_fuse_shot.shape[1:])
            freq_fuse_shot = freq_fuse_shot.mean(dim=0)
            data_shot = data_shot.contiguous().view(args.shot, args.way, *data_shot.shape[1:])
            data_shot = data_shot.mean(dim=0)

        model.module.mode = 'base'

        logits, absolute_logits = model((data_shot, data_query))
        logits_freq, absolute_logits_freq = model((freq_fuse_shot, freq_fuse_query))

        logits = logits + logits_freq

        epi_loss = F.cross_entropy(logits, label)
        absolute_loss = F.cross_entropy(absolute_logits, train_labels[k:])

        model.module.mode = 'fc'
        logits_aux = model(data_aux)
        loss_aux = F.cross_entropy(logits_aux, train_labels_aux)


        loss_aux = loss_aux + absolute_loss
        loss = args.lamb * epi_loss + loss_aux

        acc = compute_accuracy(logits, label)

        loss_meter.update(loss.item())
        acc_meter.update(acc)
        tqdm_gen.set_description(f'[train] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.acc:{acc_meter.avg():.3f} (curr:{acc:.3f})')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        # detect_grad_nan(model)
        optimizer.step()
        optimizer.zero_grad()

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()


def train_main(args):

    start_epoch =args.start_epoch
    stop_epoch = args.max_epoch
    max_acc, max_epoch = 0.0, 0
    set_seed(args.seed)

    Dataset = dataset_builder(args)
    trainset = Dataset('train', args)

    train_sampler = CategoriesSampler(trainset.label, len(trainset.data) // args.batch, args.way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=8, pin_memory=True)

    trainset_aux = Dataset('train', args)
    train_loader_aux = DataLoader(dataset=trainset_aux, batch_size=args.batch, shuffle=True, num_workers=8, pin_memory=True)

    train_loaders = {'train_loader': train_loader, 'train_loader_aux': train_loader_aux}

    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label, args.val_episode, args.way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)
    ''' fix val set for all epochs '''
    val_loader = [x for x in val_loader]

    model = DFIA(args,resnet=args.resnet).cuda()

    model = nn.DataParallel(model, device_ids=args.device_ids)
    total = sum([param.nelement() for param in model.parameters()])
    print('Number  of parameter: % .2fM' % (total / 1e6))
    if args.resume:
        resume_file = get_resume_file(args.save_path)
        print("resume_file:",resume_file)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            model.load_state_dict(tmp['params'])


    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)


    for epoch in range(start_epoch, args.max_epoch + 1):
        start_time = time.time()

        train_loss, train_acc, _ = train(epoch, model, train_loaders, optimizer, args)

        lr_scheduler.step()
        val_loss, val_acc, _ = evaluate(epoch, model ,val_loader, args, set='val')


        if val_acc > max_acc : #for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
            print(f'[ log ] *********A better model is found ({val_acc:.3f}) *********')
            max_acc, max_epoch = val_acc, epoch
            outfile = os.path.join(args.save_path, 'max_acc.pth')
            torch.save({'epoch':epoch, 'params':model.state_dict()}, outfile)

        if (epoch % args.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(args.save_path, '{:d}.pth'.format(epoch))
            torch.save({'epoch':epoch, 'params':model.state_dict()}, outfile)

        epoch_time = time.time() - start_time
        print(f'[ log ] saving @ {args.save_path}')
        print(f'[ log ] roughly {(args.max_epoch - epoch) / 3600. * epoch_time:.2f} h left\n')

    return model


def contrastive_loss( distances,label,k,margin=1.0):

    spt_labels, qry_labels = label[:k], label[k:]
    label_similarity = (qry_labels.unsqueeze(1) == spt_labels.unsqueeze(0)).float()
    positive_loss = label_similarity * distances ** 2
    negative_loss = (1 - label_similarity) * F.relu(margin - distances) ** 2

    return positive_loss.mean() + negative_loss.mean()

if __name__ == '__main__':
    args = setup_run(arg_mode='train')
    set_seed(args.seed)
    model = train_main(args)
    test_acc, test_ci = test_main(model, args)
