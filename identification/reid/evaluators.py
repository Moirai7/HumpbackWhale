from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import scipy.io as sio
import torch
import pandas as pd
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
    for i, (imgs, pids, img_labels,img_name) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs = extract_cnn_feature(model, imgs)

        #print(i, len(fnames),len(outputs),len(pids))
        for fname, output, pid in zip(img_name, outputs, img_labels):
            features[fname] = output
            labels[fname] = pid
            print(fname,features[fname])

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

        #print("+++++++++++++++")

    return features, labels


def pairwise_distance(query_features, gallery_features, query=None, gallery=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist = torch.pow(x, 2).sum(1) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist
    print("+++++++++++++++")
    print(query_features[query.Image[1]])
    x = torch.cat([query_features[f].unsqueeze(0) for f in query.Image], 0)
    y = torch.cat([gallery_features[f].unsqueeze(0) for f in gallery.Image], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(1).unsqueeze(1).expand(m, n) + \
           torch.pow(y, 2).sum(1).unsqueeze(1).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())#find (x-y)^2
    return dist

def find_top5_label(distmat, gallery=None):
    sort_dist,sort_idx = torch.sort(distmat,dim=1,descending=False)
    lable_list =[]
    for i in range(sort_dist.size(0)):
        tmp_lable_list=[]
        tmp_num= 0
        for j in sort_idx[i]:
            if  gallery.Id[j] not in tmp_lable_list:
                tmp_lable_list.append(gallery.Id[j])
                tmp_num = tmp_num + 1
                if tmp_num >= 5:
                    tmp_lable_str = ""
                    for s in tmp_lable_list:
                        tmp_lable_str = tmp_lable_str +" "+ gallery.Id[j]
                    lable_list.append(tmp_lable_str)
                    break

    #
    # top_dist,top_list = torch.topk(distmat,5,dim=1,largest=False)
    # lable_list =[gallery.Id[i].unsqueeze(0) for i in top_list]
    # lable_list = torch.cat(lable_list,dim=0)
    return lable_list



class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, query_loader, gallery_loader, query, gallery):
        print('extracting query features\n')
        query_features, query_label = extract_features(self.model, query_loader)
        print('extracting gallery features\n')
        gallery_features, gallery_label = extract_features(self.model, gallery_loader)
        distmat = pairwise_distance(query_features, gallery_features, query, gallery)
        label = find_top5_label(distmat, gallery=gallery)
        dataframe = pd.DataFrame({'Image': query.Image, 'Id': label})

        # 将DataFrame存储为csv,index表示是否显示行名，default=True
        dataframe.to_csv("~/HumpbackWhale/result.csv", index=False, sep=',')

        return find_top5_label(distmat, gallery=gallery)
