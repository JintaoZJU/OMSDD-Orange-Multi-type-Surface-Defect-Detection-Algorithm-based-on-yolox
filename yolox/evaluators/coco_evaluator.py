#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import io
import itertools
import json
import tempfile
import time
from collections import ChainMap, defaultdict
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm
import math
import warnings
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np

import torch

from yolox.data.datasets import COCO_CLASSES
from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)


def per_class_AR_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AR"], colums=6):
    per_class_AR = {}
    recalls = coco_eval.eval["recall"]
    # dimension of recalls: [TxKxAxM]
    # recall has dims (iou, cls, area range, max dets)
    assert len(class_names) == recalls.shape[1]

    for idx, name in enumerate(class_names):
        recall = recalls[0, idx, 0, -1]
        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")
        per_class_AR[name] = float(ar * 100)

    num_cols = min(colums, len(per_class_AR) * len(headers))
    result_pair = [x for pair in per_class_AR.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


def per_class_AP_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AP"], colums=6):
    AP = float(0)
    AR = float(0)
    per_class_AR = {}
    recalls = coco_eval.eval["recall"]
    # dimension of recalls: [TxKxAxM]
    # recall has dims (iou, cls, area range, max dets)
    assert len(class_names) == recalls.shape[1]

    per_class_AP = {}
    precisions = coco_eval.eval["precision"]
    # dimension of precisions: [TxRxKxAxM]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    per_class_F1 = {}

    for idx, name in enumerate(class_names):
        recall = recalls[0, idx, 0, -1]
        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")
        per_class_AR[name] = float(ar * 100)
        AR = AR + per_class_AR[name]
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[0, :, idx, 0, -1]
        precision = precision[precision > -1]

        ap = np.mean(precision) if precision.size else float("nan")
        per_class_AP[name] = float(ap * 100)
        AP = AP + per_class_AP[name]
        # per_class_f1        
        per_class_F1[name] = 2 * float(ap * 100) * float(ar *100) / (float(ap * 100) + float(ar * 100))

    num_cols = min(colums, len(per_class_AP) * len(headers))
    result_pair = [x for pair in per_class_AP.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
# 输出F1 table
    result_pair2 = [x for pair in per_class_F1.items() for x in pair]
    row_pair2 = itertools.zip_longest(*[result_pair2[i::num_cols] for i in range(num_cols)])
    table_headers2 = ["class", "F1"] * (num_cols // len(["class", "F1"]))
    table2 = tabulate(
        row_pair2, tablefmt="pipe", floatfmt=".3f", headers=table_headers2, numalign="left",
    )

    AP = AP/5
    AR = AR/5
    F1 = 2 * AP * AR / (AP + AR)
    print("F1 of all classes = ", F1)

    return table, table2, F1


class COCOEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self,
        dataloader,
        img_size: int,
        confthre: float,
        nmsthre: float,
        num_classes: int,
        testdev: bool = False,
        per_class_AP: bool = True,
        per_class_AR: bool = True,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size: image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre: confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre: IoU threshold of non-max supression ranging from 0 to 1.
            per_class_AP: Show per class AP during evalution or not. Default to False.
            per_class_AR: Show per class AR during evalution or not. Default to False.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev
        self.per_class_AP = per_class_AP
        self.per_class_AR = per_class_AR

    def evaluate(
        self, model, distributed=False, half=False, trt_file=None,
        decoder=None, test_size=None, return_outputs=False
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        output_data = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
        # #----------------- 定义变量 -------------------#
        # iouv = torch.linspace(0.5, 0.95, 10, device='cpu')
        # niou = iouv.numel()
        # confusion_matrix = ConfusionMatrix(nc=self.num_classes)
        # stats=[]
        # seen=0
        # names=["pedicel", "deformity", "base", "scarring", "canker"] #类名
        # # cocoGt = self.dataloader.dataset.coco
        # # cat_ids = list(cocoGt.cats.keys())
        # # names = [cocoGt.cats[catld]['name'] for catld in sorted(cat_ids)]
        # names_dic=dict(enumerate(names)) #类名字典
        # s = ('\n%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')    
        # save_dir = Path('./YOLOX_outputs/exps/')
        # #----------------- 定义变量 -------------------#

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)

                # skip the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                outputs = postprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre
                )
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

            data_list_elem, image_wise_data = self.convert_to_coco_format(
                outputs, info_imgs, ids, return_outputs=True)
            data_list.extend(data_list_elem)
            output_data.update(image_wise_data)
            # #----------------------- add ---------------------#
            # for _id,out in zip(ids,outputs):
            #     seen += 1
            #     gtAnn=self.dataloader.dataset.coco.imgToAnns[int(_id)]
            #     tcls=[(its['category_id'])for its in gtAnn]
            #     #---FJT add---#
            #     if out == None:
            #         if len(gtAnn)>0:
            #             stats.append((torch.zeros(0, niou, dtype = torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            #         continue
            #     #---FJT add---#
            #     # if out==None: 
            #     #     stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            #     #     continue
            #     # else:
            #     #---FJT add---#
            #     if len(gtAnn)>0:
            #     #---FJT add---#
            #         gt=torch.tensor([[(its['category_id'])]+its['clean_bbox'] for its in gtAnn])
            #         dt=out.cpu().numpy()
            #         dt[:,4]=dt[:,4]*dt[:,5]
            #         dt[:,5]=dt[:,6]
            #         dt=torch.from_numpy(np.delete(dt,-1,axis=1))#share mem
            #         confusion_matrix.process_batch(dt, gt)
            #         correct = process_batch(dt, gt, iouv)
            #         stats.append((correct, dt[:, 4], dt[:, 5], tcls))
            # #----------------------- add ---------------------#

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            output_data = gather(output_data, dst=0)
            data_list = list(itertools.chain(*data_list))
            output_data = dict(ChainMap(*output_data))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        # #----------------------- add ---------------------#
        # stats = [np.concatenate(x, 0) for x in zip(*stats)]
        # tp, fp, p, r, f1, ap, ap_class =ap_per_class(*stats, plot=True, save_dir=save_dir, names=names_dic)
        # confusion_matrix.plot(save_dir=save_dir, names=names)
        # ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        # mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        # nt = np.bincount(stats[3].astype(np.int64), minlength=self.num_classes)
        # pf = '\n%20s' + '%11i'  *2 + '%11.3g' * 4  # print format
        # s+=pf % ('all',seen, nt.sum(), mp, mr, map50, map)
        # for i, c in enumerate(ap_class):
        #     s+=pf % (names[c],seen, nt[c], p[i], r[i], ap50[i], ap[i])
        # logger.info(s)      # log出P，R，mAP50，mAP95
        # #----------------------- add ---------------------#

        if return_outputs:
            return eval_results, output_data
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids, return_outputs=False):
        data_list = []
        image_wise_data = defaultdict(dict)
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            image_wise_data.update({
                int(img_id): {
                    "bboxes": [box.numpy().tolist() for box in bboxes],
                    "scores": [score.numpy().item() for score in scores],
                    "categories": [
                        self.dataloader.dataset.class_ids[int(cls[ind])]
                        for ind in range(bboxes.shape[0])
                    ],
                }
            })

            bboxes = xyxy2xywh(bboxes)

            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)

        if return_outputs:
            return data_list, image_wise_data
        return data_list

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            if self.testdev:
                json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
                cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, "w"))
                cocoDt = cocoGt.loadRes(tmp)
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools.cocoeval import COCOeval

                logger.warning("Use standard COCOeval.")

            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            cat_ids = list(cocoGt.cats.keys())
            cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]
            if self.per_class_AP:
                AP_table, F1_table, F1 = per_class_AP_table(cocoEval, class_names=cat_names)
                info += "per class AP:\n" + AP_table + "\n"
                info += "per class F1:\n" + F1_table + "\n"
                info += "All class F1:\n" + str(F1) + "\n"
            if self.per_class_AR:
                AR_table = per_class_AR_table(cocoEval, class_names=cat_names)
                info += "per class AR:\n" + AR_table + "\n"
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info



# #-----------------------------YOLOV5画图--------------------------------#
    
# def fitness(x):
#     # Model fitness as a weighted combination of metrics
#     w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
#     return (x[:, :4] * w).sum(1)


# def smooth(y, f=0.05):
#     # Box filter of fraction f
#     nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
#     p = np.ones(nf // 2)  # ones padding
#     yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
#     return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed


# def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16):
#     """ Compute the average precision, given the recall and precision curves.
#     Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
#     # Arguments
#         tp:  True positives (nparray, nx1 or nx10).
#         conf:  Objectness value from 0-1 (nparray).
#         pred_cls:  Predicted object classes (nparray).
#         target_cls:  True object classes (nparray).
#         plot:  Plot precision-recall curve at mAP@0.5
#         save_dir:  Plot save directory
#     # Returns
#         The average precision as computed in py-faster-rcnn.
#     """

#     # Sort by objectness
#     i = np.argsort(-conf)
#     tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

#     # Find unique classes
#     unique_classes, nt = np.unique(target_cls, return_counts=True)
#     nc = unique_classes.shape[0]  # number of classes, number of detections

#     # Create Precision-Recall curve and compute AP for each class
#     px, py = np.linspace(0, 1, 1000), []  # for plotting
#     ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
#     for ci, c in enumerate(unique_classes):
#         i = pred_cls == c
#         n_l = nt[ci]  # number of labels
#         n_p = i.sum()  # number of predictions
#         if n_p == 0 or n_l == 0:
#             continue

#         # Accumulate FPs and TPs
#         fpc = (1 - tp[i]).cumsum(0)
#         tpc = tp[i].cumsum(0)

#         # Recall
#         recall = tpc / (n_l + eps)  # recall curve
#         r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

#         # Precision
#         precision = tpc / (tpc + fpc)  # precision curve
#         p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

#         # AP from recall-precision curve
#         for j in range(tp.shape[1]):
#             ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
#             if plot and j == 0:
#                 py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

#     # Compute F1 (harmonic mean of precision and recall)
#     f1 = 2 * p * r / (p + r + eps)
#     names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
#     names = dict(enumerate(names))  # to dict
#     if plot:
#         plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
#         plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
#         plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
#         plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

#     i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
#     p, r, f1 = p[:, i], r[:, i], f1[:, i]
#     tp = (r * nt).round()  # true positives
#     fp = (tp / (p + eps) - tp).round()  # false positives
#     return tp, fp, p, r, f1, ap, unique_classes.astype(int)


# def compute_ap(recall, precision):
#     """ Compute the average precision, given the recall and precision curves
#     # Arguments
#         recall:    The recall curve (list)
#         precision: The precision curve (list)
#     # Returns
#         Average precision, precision curve, recall curve
#     """

#     # Append sentinel values to beginning and end
#     mrec = np.concatenate(([0.0], recall, [1.0]))
#     mpre = np.concatenate(([1.0], precision, [0.0]))

#     # Compute the precision envelope
#     mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

#     # Integrate area under curve
#     method = 'interp'  # methods: 'continuous', 'interp'
#     if method == 'interp':
#         x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
#         ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
#     else:  # 'continuous'
#         i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
#         ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

#     return ap, mpre, mrec


# class ConfusionMatrix:
#     # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
#     def __init__(self, nc, conf=0.25, iou_thres=0.45):
#         self.matrix = np.zeros((nc + 1, nc + 1))
#         self.nc = nc  # number of classes
#         self.conf = conf
#         self.iou_thres = iou_thres

#     def process_batch(self, detections, labels):
#         """
#         Return intersection-over-union (Jaccard index) of boxes.
#         Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
#         Arguments:
#             detections (Array[N, 6]), x1, y1, x2, y2, conf, class
#             labels (Array[M, 5]), class, x1, y1, x2, y2
#         Returns:
#             None, updates confusion matrix accordingly
#         """
#         detections = detections[detections[:, 4] > self.conf]
#         gt_classes = labels[:, 0].int()
#         detection_classes = detections[:, 5].int()
#         iou = box_iou(labels[:, 1:], detections[:, :4])

#         x = torch.where(iou > self.iou_thres)
#         if x[0].shape[0]:
#             matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
#             if x[0].shape[0] > 1:
#                 matches = matches[matches[:, 2].argsort()[::-1]]
#                 matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
#                 matches = matches[matches[:, 2].argsort()[::-1]]
#                 matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
#         else:
#             matches = np.zeros((0, 3))

#         n = matches.shape[0] > 0
#         m0, m1, _ = matches.transpose().astype(int)
#         for i, gc in enumerate(gt_classes):
#             j = m0 == i
#             if n and sum(j) == 1:
#                 self.matrix[detection_classes[m1[j]], gc] += 1  # correct
#             else:
#                 self.matrix[self.nc, gc] += 1  # background FP

#         if n:
#             for i, dc in enumerate(detection_classes):
#                 if not any(m1 == i):
#                     self.matrix[dc, self.nc] += 1  # background FN

#     def matrix(self):
#         return self.matrix

#     def tp_fp(self):
#         tp = self.matrix.diagonal()  # true positives
#         fp = self.matrix.sum(1) - tp  # false positives
#         # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
#         return tp[:-1], fp[:-1]  # remove background class

#     def plot(self, normalize=True, save_dir='', names=()):
#         try:
#             import seaborn as sn

#             array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
#             array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

#             fig = plt.figure(figsize=(12, 9), tight_layout=True)
#             nc, nn = self.nc, len(names)  # number of classes, names
#             sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
#             labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
#             with warnings.catch_warnings():
#                 warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
#                 sn.heatmap(array,
#                            annot=nc < 30,
#                            annot_kws={
#                                "size": 8},
#                            cmap='Blues',
#                            fmt='.2f',
#                            square=True,
#                            vmin=0.0,
#                            xticklabels=names + ['background FP'] if labels else "auto",
#                            yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
#             fig.axes[0].set_xlabel('True')
#             fig.axes[0].set_ylabel('Predicted')
#             fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
#             plt.close()
#         except Exception as e:
#             print(f'WARNING: ConfusionMatrix plot failure: {e}')

#     def print(self):
#         for i in range(self.nc + 1):
#             print(' '.join(map(str, self.matrix[i])))


# def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
#     # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

#     # Get the coordinates of bounding boxes
#     if xywh:  # transform from xywh to xyxy
#         (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, 1), box2.chunk(4, 1)
#         w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
#         b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
#         b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
#     else:  # x1, y1, x2, y2 = box1
#         b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, 1)
#         b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, 1)
#         w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
#         w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

#     # Intersection area
#     inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
#             (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

#     # Union Area
#     union = w1 * h1 + w2 * h2 - inter + eps

#     # IoU
#     iou = inter / union
#     if CIoU or DIoU or GIoU:
#         cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
#         ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
#         if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
#             c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
#             rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
#             if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
#                 v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
#                 with torch.no_grad():
#                     alpha = v / (v - iou + (1 + eps))
#                 return iou - (rho2 / c2 + v * alpha)  # CIoU
#             return iou - rho2 / c2  # DIoU
#         c_area = cw * ch + eps  # convex area
#         return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
#     return iou  # IoU


# def box_area(box):
#     # box = xyxy(4,n)
#     return (box[2] - box[0]) * (box[3] - box[1])


# def box_iou(box1, box2, eps=1e-7):
#     # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
#     """
#     Return intersection-over-union (Jaccard index) of boxes.
#     Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
#     Arguments:
#         box1 (Tensor[N, 4])
#         box2 (Tensor[M, 4])
#     Returns:
#         iou (Tensor[N, M]): the NxM matrix containing the pairwise
#             IoU values for every element in boxes1 and boxes2
#     """

#     # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
#     (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
#     inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

#     # IoU = inter / (area1 + area2 - inter)
#     return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter + eps)


# def bbox_ioa(box1, box2, eps=1e-7):
#     """ Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
#     box1:       np.array of shape(4)
#     box2:       np.array of shape(nx4)
#     returns:    np.array of shape(n)
#     """

#     # Get the coordinates of bounding boxes
#     b1_x1, b1_y1, b1_x2, b1_y2 = box1
#     b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

#     # Intersection area
#     inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
#                  (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

#     # box2 area
#     box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

#     # Intersection over box2 area
#     return inter_area / box2_area


# def wh_iou(wh1, wh2, eps=1e-7):
#     # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
#     wh1 = wh1[:, None]  # [N,1,2]
#     wh2 = wh2[None]  # [1,M,2]
#     inter = torch.min(wh1, wh2).prod(2)  # [N,M]
#     return inter / (wh1.prod(2) + wh2.prod(2) - inter + eps)  # iou = inter / (area1 + area2 - inter)


# # Plots ----------------------------------------------------------------------------------------------------------------


# def plot_pr_curve(px, py, ap, save_dir=Path('pr_curve.png'), names=()):
#     # Precision-recall curve
#     fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
#     py = np.stack(py, axis=1)

#     if 0 < len(names) < 21:  # display per-class legend if < 21 classes
#         for i, y in enumerate(py.T):
#             ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
#     else:
#         ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

#     ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
#     ax.set_xlabel('Recall')
#     ax.set_ylabel('Precision')
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
#     fig.savefig(save_dir, dpi=250)
#     plt.close()


# def plot_mc_curve(px, py, save_dir=Path('mc_curve.png'), names=(), xlabel='Confidence', ylabel='Metric'):
#     # Metric-confidence curve
#     fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

#     if 0 < len(names) < 21:  # display per-class legend if < 21 classes
#         for i, y in enumerate(py):
#             ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
#     else:
#         ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

#     y = smooth(py.mean(0), 0.05)
#     ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
#     fig.savefig(save_dir, dpi=250)
#     plt.close()
    
# def process_batch(detections, labels, iouv):
#     """
#     Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
#     Arguments:
#         detections (Array[N, 6]), x1, y1, x2, y2, conf, class
#         labels (Array[M, 5]), class, x1, y1, x2, y2
#     Returns:
#         correct (Array[N, 10]), for 10 IoU levels
#     """
#     correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
#     iou = box_iou(labels[:, 1:], detections[:, :4])
#     correct_class = labels[:, 0:1] == detections[:, 5]
#     for i in range(len(iouv)):
#         x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
#         if x[0].shape[0]:
#             matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
#             if x[0].shape[0] > 1:
#                 matches = matches[matches[:, 2].argsort()[::-1]]
#                 matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
#                 # matches = matches[matches[:, 2].argsort()[::-1]]
#                 matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
#             correct[matches[:, 1].astype(int), i] = True
#     return torch.tensor(correct, dtype=torch.bool, device=iouv.device)