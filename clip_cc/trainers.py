from __future__ import print_function, absolute_import

import time
from torch.cuda import amp
from .utils.meters import AverageMeter
import torch


class VITFP16(object):
    def __init__(self, encoder, memory=None):
        super(VITFP16, self).__init__()
        self.encoder = encoder
        self.memory = memory

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400 ):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()
        # amp fp16 training
        scaler = amp.GradScaler()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)
            with amp.autocast(enabled=True):
                # process inputs
                inputs, labels, indexes = self._parse_data(inputs)
                # forward
                f_out = self.encoder(inputs)
                loss= self.memory(f_out,labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      .format(epoch + 1, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              losses.val, losses.avg
                              ))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()