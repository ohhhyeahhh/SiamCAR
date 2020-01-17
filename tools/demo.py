from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
sys.path.append('../')

import argparse
import cv2
import torch
from glob import glob

from siamcar.core.config import cfg
from siamcar.models.model_builder import ModelBuilder
from siamcar.tracker.siamcar_tracker import SiamCARTracker
from siamcar.utils.model_load import load_pretrain

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='SiamCAR demo')
parser.add_argument('--config', type=str, default='../siamcar/config.yaml', help='config file')
parser.add_argument('--snapshot', type=str, default='./snapshot/general_model.pth', help='model name')
parser.add_argument('--video_name', default='../testing_dataset/Biker', type=str, help='videos or image files')
parser.add_argument('--hp_search', default='OTB', type=str, help='hp_search parameters')
args = parser.parse_args()


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)

        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).eval().to(device)

    # build tracker
    tracker = SiamCARTracker(model, cfg.TRACK)

    # hp_search
    if 'GOT' in args.hp_search:
        hp = {'lr': cfg.HP_SEARCH.GOT[0], 'pk': cfg.HP_SEARCH.GOT[1], 'w_lr': cfg.HP_SEARCH.GOT[2]}
    elif 'UAV' in args.hp_search:
        hp = {'lr': cfg.HP_SEARCH.UAV123[0], 'pk': cfg.HP_SEARCH.UAV123[1], 'w_lr': cfg.HP_SEARCH.UAV123[2]}
    elif 'OTB' in args.hp_search:
        hp = {'lr': cfg.HP_SEARCH.OTB[0], 'pk': cfg.HP_SEARCH.OTB[1], 'w_lr': cfg.HP_SEARCH.OTB[2]}
    elif 'VOT2019' in args.hp_search:
        hp = {'lr': cfg.HP_SEARCH.VOT2019[0], 'pk': cfg.HP_SEARCH.VOT2019[1], 'w_lr': cfg.HP_SEARCH.VOT2019[2]}
    elif 'LaSOT' in args.hp_search:
        hp = {'lr': cfg.HP_SEARCH.LaSOT[0], 'pk': cfg.HP_SEARCH.LaSOT[1], 'w_lr': cfg.HP_SEARCH.LaSOT[2]}
    else:
        raise KeyError("Non-existent hp parameters: {}".format(args.hp_search))

    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    for frame in get_frames(args.video_name):
        if first_frame:
            try:
                init_rect = cv2.selectROI(video_name, frame, False, False)
            except:
                exit()
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            outputs = tracker.track(frame, hp)
            bbox = list(map(int, outputs['bbox']))
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                          (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                          (0, 255, 0), 3)
            cv2.imshow(video_name, frame)
            cv2.waitKey(40)


if __name__ == '__main__':
    main()
