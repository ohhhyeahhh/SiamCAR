# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F
import cv2

from siamcar.core.config import cfg
from siamcar.tracker.base_tracker import SiameseTracker


class SiamCARTracker(SiameseTracker):
    def __init__(self, model, cfg):
        super(SiamCARTracker, self).__init__()
        self.cfg = cfg
        self.score_size = cfg.SCORE_SIZE
        hanning = np.hanning(self.score_size)
        self.window = np.outer(hanning, hanning)
        self.score_size_up = cfg.UPSIZE
        self.model = model
        self.model.eval()

    def _convert_score(self, score):
        score = F.softmax(score[:, :, :, :], dim=1).data[:, 1, :, :].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2, bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.crop_template(img)
        self.model.template(z_crop)

    def crop_template(self, img):
        # calculate z crop size
        w_z, h_z = self.size + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        z = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        return z

    def crop_search_region(self, img):
        w_z, h_z = self.size + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)

        self.scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
        return x

    # get bbox
    def get_bbox(self, p_coor, cen, bboxes):
        max_r, max_c = p_coor[0], p_coor[1]
        bbox = bboxes[:, max_r, max_c]
        l, t, r, b = bbox
        cen_value = cen[max_r, max_c]
        return np.array([l, t, r, b]), cen_value

    def change(self, r):
        return np.maximum(r, 1. / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5
        return np.sqrt((w + pad) * (h + pad))

    def cal_penalty(self, bboxes, size_ori, penalty_lk):
        bboxes_w = bboxes[0, :, :] + bboxes[2, :, :]
        bboxes_h = bboxes[1, :, :] + bboxes[3, :, :]
        s_c = self.change(self.sz(bboxes_w, bboxes_h) / self.sz(size_ori[0]*self.scale_z, size_ori[1]*self.scale_z))
        r_c = self.change((size_ori[0] / size_ori[1]) / (bboxes_w / bboxes_h))
        penalty = np.exp(-(r_c * s_c - 1) * penalty_lk)
        return penalty

    def select_points(self, max_r, max_c, res_map):
        value_list, point_list = [], []
        num = 0
        res_map_shape = res_map.shape
        for i in range(-1, 2):
            for j in range(-1, 2):
                def indexofMin(arr):
                    minindex = 0
                    currentindex = 1
                    while currentindex < len(arr):
                        if arr[currentindex] < arr[minindex]:
                            minindex = currentindex
                        currentindex += 1
                    return minindex

                num += 1
                in_max_r, in_max_c = max(0, min(24, max_r + i)), max(0, min(24, max_c + j))

                if in_max_c < 0 or in_max_c >= res_map_shape[0] or \
                        in_max_r < 0 or in_max_r >= res_map_shape[0]:
                    continue
                if num <= 2:
                    value_list.append(res_map[in_max_r, in_max_c])
                    p_cool = np.array([in_max_r, in_max_c])
                    point_list.append(p_cool)
                else:
                    loc = indexofMin(value_list)
                    if res_map[in_max_r, in_max_c] > min(value_list):
                        value_list[loc] = res_map[in_max_r, in_max_c]
                        p_cool = np.array([in_max_r, in_max_c])
                        point_list[loc] = p_cool

        return point_list

    def corse_location(self, max_r_up_c, max_c_up_c, scale_resmap, bboxes):
        def region_m(size):
            if size > 56:
                size = 56
            elif size < 12:
                size = 12
            else:
                size = size
            return size
        max_r_c = int(round(max_r_up_c/scale_resmap))
        max_c_c = int(round(max_c_up_c/scale_resmap))
        bbox_region = bboxes[:, max_r_c, max_c_c]
        l_region = int(min(max_r_up_c, region_m(bbox_region[0])))
        t_region = int(min(max_c_up_c, region_m(bbox_region[1])))
        r_region = int(min(193-max_r_up_c, region_m(bbox_region[2])))
        b_region = int(min(193-max_c_up_c, region_m(bbox_region[3])))
        return np.array([l_region, t_region, r_region, b_region])

    def average_location(self, max_r, max_c, p_score, score, bboxes):
        # select some points to calculate bbox
        point_list, value_list = [], []
        if self.cfg.ave_reg:
            point_list = self.select_points(max_r, max_c, p_score)
        else:
            p_cool = np.array([max_r, max_c])
            point_list.append(p_cool)

        # average location
        ave_bbox = 0
        for i, point in enumerate(point_list):
            bbox, s_value = self.get_bbox(point, score, bboxes)
            pw_bbox = bbox*s_value
            ave_bbox += pw_bbox
            value_list.append(s_value)
        ave_bbox = ave_bbox/np.sum(np.array(value_list))
        return ave_bbox

    def track(self, img, hp):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        x_crop = self.crop_search_region(img)
        outputs = self.model.track(x_crop)

        score = self._convert_score(outputs['cls'])
        score = score.squeeze()

        cen = outputs['cen'].data.cpu().numpy()
        cen = cen.squeeze()

        bboxes = outputs['loc'].data.cpu().numpy()
        bboxes = bboxes.squeeze()

        penalty = self.cal_penalty(bboxes, self.size, hp['pk'])
        p_score = penalty * score
        if self.cfg.hanming:
            hp_score = p_score*(1-hp['w_lr']) + self.window * hp['w_lr']
        else:
            hp_score = p_score

        hp_score_up = cv2.resize(hp_score, (193, 193), interpolation=cv2.INTER_CUBIC)
        score_up = cv2.resize(score, (193, 193), interpolation=cv2.INTER_CUBIC)
        cen_up = cv2.resize(cen, (193, 193), interpolation=cv2.INTER_CUBIC)
        scale_resmap = 193 / 25
        res_map_up = score_up * cen_up

        max_r_up_c, max_c_up_c = np.unravel_index(hp_score_up.argmax(), hp_score_up.shape)

        # corse location
        ltrb_region = self.corse_location(max_r_up_c, max_c_up_c, scale_resmap, bboxes)
        res_map_up[max_r_up_c-ltrb_region[0]:max_r_up_c+ltrb_region[2]+1, \
                    max_c_up_c-ltrb_region[1]:max_c_up_c+ltrb_region[3]+1] += 100

        # accurate location
        max_r_up, max_c_up = np.unravel_index(res_map_up.argmax(), res_map_up.shape)
        max_r, max_c = int(round(max_r_up/scale_resmap)), int(round(max_c_up/scale_resmap))
        max_r_up += 31
        max_c_up += 31

        # average location
        ave_bbox = self.average_location(max_r, max_c, p_score, score, bboxes)

        p_cool_s = np.array([max_r_up, max_c_up])
        disp = p_cool_s - (np.array([255, 255]) - 1.) / 2.

        l, t, r, b = ave_bbox

        ave_w = (l + r) / self.scale_z
        ave_h = (t + b) / self.scale_z

        disp_ori = disp / self.scale_z
        ave_cx = disp_ori[1] + self.center_pos[0]
        ave_cy = disp_ori[0] + self.center_pos[1]

        s_c = self.change(self.sz(ave_w, ave_h) / self.sz(self.size[0]*self.scale_z, self.size[1]*self.scale_z))
        r_c = self.change((self.size[0] / self.size[1]) / (ave_w / ave_h))
        penalty = np.exp(-(r_c * s_c - 1) * hp['pk'])
        lr = penalty * score[max_r, max_c] * hp['lr']
        width = lr * ave_w + (1-lr) * self.size[0]
        height = lr * ave_h + (1-lr) * self.size[1]

        # clip boundary
        cx, cy, width, height = self._bbox_clip(ave_cx, ave_cy, width, height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2, cy - height / 2, width, height]

        return {
                'bbox': bbox,
               }
