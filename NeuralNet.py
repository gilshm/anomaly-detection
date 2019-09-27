from models.wide_resnet import WideResNet
import torch
import time
import Config as cfg
import torch.nn.functional as F
import matplotlib.pyplot as plt

DEPTH = 28


class NeuralNet:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            print('WARNING: Found no valid GPU device - Running on CPU')

        self.model = WideResNet(DEPTH, cfg.NUM_TRANS)
        self.model.cuda(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().cuda(self.device)
        self.optimizer = None

    def test(self, test_gen):
        self.model.eval()

        batch_time = self.AverageMeter('Time', ':6.3f')
        losses = self.AverageMeter('Loss', ':.4e')
        top1 = self.AverageMeter('Acc@1', ':6.2f')
        progress = self.ProgressMeter(len(test_gen), batch_time, losses, top1, prefix='Test: ')

        # switch to evaluate mode
        self.model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (input, target, _) in enumerate(test_gen):
                input = input.cuda(self.device, non_blocking=True)
                target = target.cuda(self.device, non_blocking=True)

                # Compute output
                output, _ = self.model([input,target])
                loss = self.criterion(output, target)

                # Measure accuracy and record loss
                acc1 = self._accuracy(output, target, topk=(1))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1[0].cpu().detach().item(), input.size(0))

                # Measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # Print to screen
                if i % 100 == 0:
                    progress.print(i)

            # TODO: this should also be done with the ProgressMeter
            print(' * Acc@1 {top1.avg:.3f}'
                  .format(top1=top1))

        return top1.avg

    def train(self, train_gen, test_gen, epochs, lr=0.0001, lr_plan=None, momentum=0.9, wd=5e-4):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        #self.optimizer = torch.optim.Adam(self.model.parameters())

        for epoch in range(epochs):
            self._adjust_lr_rate(self.optimizer, epoch, lr_plan)
            print("=> Training (specific label)")
            self._train_step(train_gen, epoch, self.optimizer)
            print("=> Validation (entire dataset)")
            self.test(test_gen)

    def _train_step(self, train_gen, epoch, optimizer):
        self.model.train()

        batch_time = self.AverageMeter('Time', ':6.3f')
        data_time = self.AverageMeter('Data', ':6.3f')
        losses = self.AverageMeter('Loss', ':.4e')
        top1 = self.AverageMeter('Acc@1', ':6.2f')
        progress = self.ProgressMeter(len(train_gen), batch_time, data_time, losses, top1,
                                      prefix="Epoch: [{}]".format(epoch))

        end = time.time()
        for i, (input, target, _) in enumerate(train_gen):
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.cuda(self.device, non_blocking=True)
            target = target.cuda(self.device, non_blocking=True)

            # Compute output
            output, trans_out = self.model([input, target])
            loss = self.criterion(output, target)

            # measure accuracy and record loss
            acc1 = self._accuracy(output, target, topk=(1))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0].cpu().detach().item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                progress.print(i)

    def evaluate(self, eval_gen):
        # switch to evaluate mode
        self.model.eval()

        score_func_list = []
        labels_list = []
        with torch.no_grad():

            for i, (input, target, labels) in enumerate(eval_gen):
                input = input.cuda(self.device, non_blocking=True)
                #Target- the transforamation class
                target = target.cuda(self.device, non_blocking=True)
                #The true label
                labels = labels[[cfg.NUM_TRANS * x for x in range(len(labels) // cfg.NUM_TRANS)]].cuda(self.device, non_blocking=True)

                # Compute output
                # #TODO: Rewrite this code section, can be more efficient
                output_SM = self.model([input, target])

                target_mat = torch.zeros_like(output_SM[0])
                target_mat[range(output_SM[0].shape[0]),target] = 1
                target_SM = (target_mat * output_SM[0]).sum(dim=1).view(-1,cfg.NUM_TRANS).sum(dim=1)

                score_func_list.append(1/cfg.NUM_TRANS * target_SM)
                labels_list.append(labels)

        return torch.cat(score_func_list), torch.cat(labels_list)

    def _adjust_lr_rate(self, optimizer, epoch, lr_dict):
        if lr_dict is None:
            return

        for key, value in lr_dict.items():
            if epoch == key:
                print("=> New learning rate set of {}".format(value))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = value

    def summary(self, x_size, print_it=True):
        return self.model.summary(x_size, print_it=print_it)

    def print_weights(self):
        self.model.print_weights()

    @staticmethod
    def _accuracy(output, target, topk=(1)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = 1
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []

            correct_k = correct[0].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

            return res

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
        def __init__(self, num_batches, *meters, prefix=""):
            self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
            self.meters = meters
            self.prefix = prefix

        def print(self, batch):
            entries = [self.prefix + self.batch_fmtstr.format(batch)]
            entries += [str(meter) for meter in self.meters]
            print('\t'.join(entries))

        def _get_batch_fmtstr(self, num_batches):
            num_digits = len(str(num_batches // 1))
            fmt = '{:' + str(num_digits) + 'd}'
            return '[' + fmt + '/' + fmt.format(num_batches) + ']'

