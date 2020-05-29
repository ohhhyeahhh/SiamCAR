#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import json
from os.path import join, exists
import os
import pandas as pd
dataset_path = './data'

def parse_and_sched(dl_dir='.'):
    # For each of the two datasets
    f = open('./train_id.txt', 'r')
    videos = f.readlines()
    f.close()
    n_videos = len(videos)
    js = {}
    for idx,video in enumerate(videos):
        print('{}/{}'.format(idx,n_videos))
        video = video.strip()
        class_name = video.split('-')[0]
        class_path = join(dataset_path, class_name)
        gt_path = join(class_path, video, 'groundtruth.txt')
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
            x1 = bbox[0]
            y1 = bbox[1]
            w = bbox[2]
            h = bbox[3]
            if x1 < 0 or y1 < 0 or w <= 0 or h <= 0:
                continue

            if video not in js:
                js[video] = {}
            if obj not in js[video]:
                js[video][obj] = {}
            js[video][obj][frame] = bbox
    json.dump(js, open('train.json', 'w'), indent=4, sort_keys=True)
    js = {}
    json.dump(js, open('val.json', 'w'), indent=4, sort_keys=True)
    print('done')


if __name__ == '__main__':
    parse_and_sched()
