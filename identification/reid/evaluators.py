from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import scipy.io as sio
import torch

from .evaluation_metrics import cmc, mean_ap
from .feature_extraction import extract_cnn_feature
from .utils.meters import AverageMeter


def extract_features(model, data_loader, print_freq=10):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    for i, (imgs, pids, fnames) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs = extract_cnn_feature(model, imgs)
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    return features, labels


def pairwise_distance(query_features, gallery_features, query=None, gallery=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist = torch.pow(x, 2).sum(1) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist

    x = torch.cat([query_features[f].unsqueeze(0) for _, _, f in query], 0)
    y = torch.cat([gallery_features[f].unsqueeze(0) for _, _, f in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(1).unsqueeze(1).expand(m, n) + \
           torch.pow(y, 2).sum(1).unsqueeze(1).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())#find (x-y)^2
    return dist

def find_top5_label(distmat, gallery=None):
    top_dist,top_list = torch.topk(distmat,5,dim=1,largest=False)
    lable_list =[gallery.Id[i].unsqueeze(0) for i in top_list]
    lable_list = torch.cat(lable_list,dim=0)
    return lable_list



class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, query_loader, gallery_loader, query, gallery):
        print('extracting query features\n')
        query_features, _ = extract_features(self.model, query_loader)
        print('extracting gallery features\n')
        gallery_features, _ = extract_features(self.model, gallery_loader)
        distmat = pairwise_distance(query_features, gallery_features, query, gallery)
        return find_top5_label(distmat, gallery=gallery)
