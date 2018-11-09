#!/usr/bin/env python


"""Generates bottom-up-attention features as a h5 file. """

# Example:
# ./tools/generate_tsv.py --gpu 0,1,2,3,4,5,6,7 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end/test.prototxt --out test2014_resnet101_faster_rcnn_genome.tsv --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --split coco_test2014

import h5py
import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect, _get_blobs
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

import caffe
import argparse
import pprint
import time, os, sys
import base64
import numpy as np
import cv2
import csv
from multiprocessing import Process
import random
import json

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']


# keys: image_features, image_bb, spatial_features, image_ids (lexicograph)

def coco_id_to_filename(id, split, ext='.jpg'):
    if split == 'train' or split == 'val':
        year = '2014'
    elif split == 'all':
        year = ''
    else:
        year = '2015'
    return 'COCO_' + split + year + "_" + str(id).rjust(12, '0') + ext


def clevr_index_to_filename(index, split, subsplit='', ext='.png'):
    return 'CLEVR_' + split + subsplit + '_' + str(index).rjust(6, '0') + ext


def coco_filename_to_id(filename):
    return int(filename.split("_")[2].split(".")[0])


def load_id_file_list(dataroot, dataset, split, subsplit):
    ''' Load a list of (path,image_id tuples). Assumes all the images are in the same directory (useful for different subsets on same set of images)'''
    id_file_list = []
    with open(os.path.join(dataroot, 'image_ids', split + '_image_ids.json')) as f:
        image_ids = json.load(f)['image_ids']
        for image_id in image_ids:
            if dataset.lower() in ['vqa2', 'tdiuc', 'cvqa', 'natural_vqa']:
                filename = coco_id_to_filename(int(image_id), split)
            elif dataset.lower() in ['clevr', 'clevr_humans', 'clevr-humans', 'clevr-cogent-a', 'clevr-cogent-b']:
                filename = clevr_index_to_filename(image_id, split, subsplit)
            filepath = os.path.join(dataroot, 'images', split, filename)
            id_file_list.append((filepath, image_id))
    return id_file_list


def get_detections_from_im(net, im_file, image_id, conf_thresh=0.2):
    im = cv2.imread(im_file)
    scores, boxes, attr_scores, rel_scores = im_detect(net, im)

    # Keep the original boxes, don't worry about the regresssion bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs['cls_prob'].data
    pool5 = net.blobs['pool5_flat'].data
    print("res5c.shape: ", np.array(net.blobs['res5c'].data).shape)
    print("res5c_relu.shape: ", np.array(net.blobs['res5c'].data).shape)
    print("pool5.shape: ", np.array(net.blobs['pool5'].data).shape)
    print("pool5_flat.shape: ", np.array(net.blobs['pool5_flat'].data).shape)
    # Keep only the best detections
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1, cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, cfg.TEST.NMS))
        max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < args.min_boxes:
        keep_boxes = np.argsort(max_conf)[::-1][:args.min_boxes]
    elif len(keep_boxes) > args.max_boxes:
        keep_boxes = np.argsort(max_conf)[::-1][:args.max_boxes]

    return {
        'image_id': image_id,
        'image_h': np.size(im, 0),
        'image_w': np.size(im, 1),
        'num_boxes': len(keep_boxes),
        'boxes': base64.b64encode(cls_boxes[keep_boxes]),
        'features': base64.b64encode(pool5[keep_boxes])
    }


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id(s) to use',
                        default='0', type=str)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to use',
                        default=None, type=str)
    parser.add_argument('--out', dest='outfile',
                        help='output filepath',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--split', dest='split',
                        help='train/val/test/all', type=str)
    parser.add_argument('--subsplit', type=str, default='', required=False)
    parser.add_argument('--dataset', help='CLEVR/VQA2/TDIUC/NATURAL_VQA etc', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--data_root',
                        help='Directory containing the data', default=None)
    parser.add_argument('--min_boxes', help='Minimum # of boxes to extract features', type=int)
    parser.add_argument('--max_boxes', help='Maximum # of boxes to extract features', type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def generate_h5(args, model, outfile):
    data_file = args.dataroot + '/faster_rcnn/{}.hdf5'.format(args.split)
    h = h5py.File(data_file, "w")

    with open(args.dataroot + '/image_ids/{}_image_ids.json'.format(args.split), 'r') as f:
        img_ids = json.load(f)['image_ids']

    img_features = h.create_dataset(
        'image_features', (len(img_ids), args.num_fixed_boxes, args.feature_length), 'f')
    img_bb = h.create_dataset(
        'image_bb', (len(img_ids), args.num_fixed_boxes, 4), 'f')
    spatial_img_features = h.create_dataset(
        'spatial_features', (len(img_ids), args.spatial_features, 6), 'f')





def generate_tsv(prototxt, weights, img_id_file, outfile):
    # First check if file exists, and if it is complete
    for file_path, img_id in img_id_file:




        with open(outfile, 'a') as tsvfile:
            writer = csv.DictWriter(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
            _t = {'misc': Timer()}
            count = 0
            for im_file, image_id in img_id_file:
                if int(image_id) in missing:
                    _t['misc'].tic()
                    writer.writerow(get_detections_from_im(net, im_file, image_id))
                    _t['misc'].toc()
                    if (count % 100) == 0:
                        print('GPU {:d}: {:d}/{:d} {:.3f}s (projected finish: {:.2f} hours)'.format(gpu_id, count + 1,
                                                                                                    len(missing), _t[
                                                                                                        'misc'].average_time,
                                                                                                    _t[
                                                                                                        'misc'].average_time * (
                                                                                                            len(
                                                                                                                missing) - count) / 3600))
                    count += 1


def merge_tsvs():
    test = ['/work/data/tsv/test2015/resnet101_faster_rcnn_final_test.tsv.%d' % i for i in range(8)]

    outfile = '/work/data/tsv/merged.tsv'
    with open(outfile, 'a') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)

        for infile in test:
            with open(infile) as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
                for item in reader:
                    try:
                        writer.writerow(item)
                    except Exception as e:
                        print(e)


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    gpus = [int(i) for i in gpu_list]

    image_ids = load_id_file_list(args.data_root, args.dataset, args.split, args.subsplit)

    procs = []

    for i, gpu_id in enumerate(gpus):
        outfile = '%s.%d' % (args.outfile, gpu_id)
        p = Process(target=generate_tsv,
                    args=(gpu_id, args.prototxt, args.caffemodel, image_ids[i], outfile))
        p.daemon = True
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
