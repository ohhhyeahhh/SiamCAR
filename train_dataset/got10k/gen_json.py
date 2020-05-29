#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import json
from os.path import join, exists
import os
import pandas as pd

dataset_path = 'data'
train_sets = ['GOT-10k_Train_split_01','GOT-10k_Train_split_02','GOT-10k_Train_split_03','GOT-10k_Train_split_04',
            'GOT-10k_Train_split_05','GOT-10k_Train_split_06','GOT-10k_Train_split_07','GOT-10k_Train_split_08',
            'GOT-10k_Train_split_09','GOT-10k_Train_split_10','GOT-10k_Train_split_11','GOT-10k_Train_split_12',
            'GOT-10k_Train_split_13','GOT-10k_Train_split_14','GOT-10k_Train_split_15','GOT-10k_Train_split_16',
            'GOT-10k_Train_split_17','GOT-10k_Train_split_18','GOT-10k_Train_split_19']
val_set = ['val']
d_sets = {'videos_val':val_set,'videos_train':train_sets}
# videos_val = ['MOT17-02-DPM']
# videos_train = ['MOT17-04-DPM','MOT17-05-DPM','MOT17-09-DPM','MOT17-11-DPM','MOT17-13-DPM']
# d_sets = {'videos_val':videos_val,'videos_train':videos_train}

def parse_and_sched(dl_dir='.'):
    # For each of the two datasets
    js = {}
    for d_set in d_sets:
        for dataset in d_sets[d_set]:
            videos = os.listdir(os.path.join(dataset_path,dataset))
            for video in videos:
                if video == 'list.txt':
                    continue
                video = dataset+'/'+video
                gt_path = join(dataset_path, video, 'groundtruth.txt')
                f = open(gt_path, 'r')
                groundtruth = f.readlines()
                f.close()
                for idx, gt_line in enumerate(groundtruth):
                    gt_image = gt_line.strip().split(',')
                    frame = '%06d' % (int(idx))
                    obj = '%02d' % (int(0))
                    bbox = [int(float(gt_image[0])), int(float(gt_image[1])),
                            int(float(gt_image[0])) + int(float(gt_image[2])),
                            int(float(gt_image[1])) + int(float(gt_image[3]))]  # xmin,ymin,xmax,ymax

                    if video not in js:
                        js[video] = {}
                    if obj not in js[video]:
                        js[video][obj] = {}
                    js[video][obj][frame] = bbox
        if 'videos_val' == d_set:
            json.dump(js, open('val.json', 'w'), indent=4, sort_keys=True)
        else:
            json.dump(js, open('train.json', 'w'), indent=4, sort_keys=True)
        js = {}

        print(d_set+': All videos downloaded' )


if __name__ == '__main__':
    parse_and_sched()
