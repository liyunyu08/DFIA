import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from common.meter import Meter
from common.utils import compute_accuracy, load_model, setup_run, by
from models.dataloader.samplers import CategoriesSampler
from models.dataloader.data_utils import dataset_builder
from models.base_model import DFIA



def evaluate(epoch, model, loader, args=None, set='val'):
    model.eval()

    loss_meter = Meter()
    acc_meter = Meter()

    label = torch.arange(args.way).repeat(args.query).cuda()
    k = args.way * args.shot

    tqdm_gen = tqdm.tqdm(loader)

    with torch.no_grad():
        for i, (data, labels) in enumerate(tqdm_gen, 1):
            data = data.cuda()
            data, train_labels = data.cuda(), label.cuda()

            # model.module.mode = 'high_pass'
            # data = model(data)

            model.module.mode = 'encoder'
            freq_fuse, spa = model(data,Re=True)
            freq_fuse_shot, freq_fuse_query = freq_fuse[:k], freq_fuse[k:]
            data_shot, data_query = spa[:k], spa[k:]

            if args.shot > 1:
                freq_fuse_shot = freq_fuse_shot.contiguous().view(args.shot, args.way, *freq_fuse_shot.shape[1:])
                freq_fuse_shot = freq_fuse_shot.mean(dim=0)
                data_shot = data_shot.contiguous().view(args.shot, args.way, *data_shot.shape[1:])
                data_shot = data_shot.mean(dim=0)

            model.module.mode = 'base'
            logits = model((data_shot, data_query))
            logits_low = model((freq_fuse_shot, freq_fuse_query))
            logits = (logits + logits_low)

            loss = F.cross_entropy(logits, label)
            acc = compute_accuracy(logits, label)

            loss_meter.update(loss.item())
            acc_meter.update(acc)
            tqdm_gen.set_description(
                f'[{set:^5}] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.acc:{by(acc_meter.avg())} (curr:{acc:.3f})')

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()



def test_main(model, args):
    ''' load model '''
    model = load_model(model, os.path.join(args.save_path, 'max_acc.pth'))


    ''' define test dataset '''
    Dataset = dataset_builder(args)
    test_set = Dataset('test', args)
    sampler = CategoriesSampler(test_set.label, args.test_episode, args.way, args.shot + args.query)
    test_loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)

    ''' evaluate the model with the dataset '''
    _, test_acc, test_ci = evaluate("best", model, test_loader, args, set='test')
    print(f'[final] epo:{"best":>3} | {by(test_acc)} ± {test_ci:.2f}')

    return test_acc, test_ci


if __name__ == '__main__':
    args = setup_run(arg_mode='test')

    ''' define model '''
    model = DFIA(args).cuda()
    model = nn.DataParallel(model, device_ids=args.device_ids)
    test_main(model, args)
