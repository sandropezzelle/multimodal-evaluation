#!/usr/bin/env python
# coding: utf-8

"""
code by Sandro Pezzelle
University of Amsterdam, March 2021
adapted from: https://github.com/airsplay/py-bottom-up-attention/blob/master/demo/demo_feature_extraction_attr.ipynb
"""

import os
import io

import detectron2

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference_single_image

# import some common libraries
import numpy as np
import cv2
import torch

# import libraries added by SP
import base64
import csv
import sys

csv.field_size_limit(sys.maxsize)

"""
set some parameters
"""
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]

img_path = '/project/dmg_data/vist/picked_images/'
out_path = '/project/dmg_data/MMdata/vist-imgs-features.tsv'
out_tsv = open(out_path, 'w', newline='')
writer = csv.DictWriter(out_tsv, delimiter='\t', fieldnames=FIELDNAMES)
NUM_OBJECTS = 36
datatype = str(img_path).split('/')[-3]
if datatype == 'vist':
    mykey = 'v'
elif datatype == 'coco2017':
    mykey = 'c'

# Load VG Classes
data_path = 'data/genome/1600-400-20'

vg_classes = []
with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        vg_classes.append(object.split(',')[0].lower().strip())

vg_attrs = []
with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
    for object in f.readlines():
        vg_attrs.append(object.split(',')[0].lower().strip())

MetadataCatalog.get("vg").thing_classes = vg_classes
MetadataCatalog.get("vg").attr_classes = vg_attrs

cfg = get_cfg()
cfg.merge_from_file("../configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml")
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
# VG Weight
cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl"
predictor = DefaultPredictor(cfg)


def doit(raw_image):
    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]
        # print("Original image size: ", (raw_height, raw_width))

        # Preprocessing
        image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        # print("Transformed image size: ", image.shape[:2])
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)

        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)

        # Generate proposals with RPN
        proposals, _ = predictor.model.proposal_generator(images, features, None)
        proposal = proposals[0]
        # print('Proposal Boxes size:', proposal.proposal_boxes.tensor.shape)

        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        # print('Pooled features size:', feature_pooled.shape)

        # Predict classes and boxes for each proposal.
        pred_class_logits, pred_attr_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(
            feature_pooled)
        outputs = FastRCNNOutputs(
            predictor.model.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            predictor.model.roi_heads.smooth_l1_beta,
        )
        probs = outputs.predict_probs()[0]
        boxes = outputs.predict_boxes()[0]

        attr_prob = pred_attr_logits[..., :-1].softmax(-1)
        max_attr_prob, max_attr_label = attr_prob.max(-1)

        # Note: BUTD uses raw RoI predictions,
        #       we use the predicted boxes instead.
        # boxes = proposal_boxes[0].tensor

        # NMS
        for nms_thresh in np.arange(0.5, 1.0, 0.1):
            instances, ids = fast_rcnn_inference_single_image(
                boxes, probs, image.shape[1:],
                score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=NUM_OBJECTS
            )
            if len(ids) == NUM_OBJECTS:
                break

        instances = detector_postprocess(instances, raw_height, raw_width)
        roi_features = feature_pooled[ids].detach()
        max_attr_prob = max_attr_prob[ids].detach()
        max_attr_label = max_attr_label[ids].detach()
        instances.attr_scores = max_attr_prob
        instances.attr_classes = max_attr_label

        # print(instances)

        """
        We output four things: instances, roi_features,
        """
        return instances, roi_features, raw_height, raw_width

c=0
for image in os.listdir(img_path):
    data = {}
    if c%100==0:
        print(c)
    img_id = str(image).split('/')[-1].split('.')[0]
    img_id_enc = mykey+'_'+str(img_id) # FIELD 0 str
    image_path = str(img_path+image)
    # print(image_path)
    im = cv2.imread(image_path)

    if im is None:
        print(c,image,'image corrupted/missing')
        continue

    instances, features, img_h, img_w = doit(im)

    pred = instances.to('cpu')
    img_h_enc = str(img_h) # FIELD 1 str
    img_w_enc = str(img_w) # FIELD 2 str
    objects_id = pred.pred_classes.detach().numpy()
    objects_id_enc = base64.b64encode(objects_id) # FIELD 3 int64
    # print(np.frombuffer(base64.b64decode(objects_id_enc),dtype=np.int64))
    objects_conf = pred.scores.detach().numpy()
    objects_conf_enc = base64.b64encode(objects_conf) # FIELD 4 float32
    attrs_id = pred.attr_classes.detach().numpy()
    attrs_id_enc = base64.b64encode(attrs_id) # FIELD 5 int64
    attrs_conf = pred.attr_scores.detach().numpy()
    attrs_conf_enc = base64.b64encode(attrs_conf) # FIELD 6 float32
    num_boxes = len(pred.pred_boxes)
    num_boxes_enc = str(num_boxes) # FIELD 7 str
    myb = []
    myb = np.array(myb)
    for el in pred.pred_boxes:
        for ell in el.detach().numpy():
            myb = np.concatenate((myb,ell),axis=None)
    boxes = np.float32(myb)
    boxes_enc = base64.b64encode(boxes) # FIELD 8 float32
    myf = []
    myf = np.array(myf)
    ft = features.cpu().numpy()
    for el in ft:
        myf = np.concatenate((myf, el), axis=None)
    features_e = np.float32(myf)
    features_e_enc = base64.b64encode(features_e) # FIELD 9 float32

    data = {
    "img_id": img_id_enc,
    "img_h": img_h_enc,
    "img_w": img_w_enc,
    "objects_id": objects_id_enc.decode("utf-8"),
    "objects_conf": objects_conf_enc.decode("utf-8"),
    "attrs_id": attrs_id_enc.decode("utf-8"),
    "attrs_conf": attrs_conf_enc.decode("utf-8"),
    "num_boxes": num_boxes_enc,
    "boxes": boxes_enc.decode("utf-8"),
    "features": features_e_enc.decode("utf-8")
    }

    if data is None:
        print('Data is empty: not saved!')
    else:
        writer.writerow(data)
        c+=1

out_tsv.close()
print('Extracted r-cnn features and saved them to .tsv!')

