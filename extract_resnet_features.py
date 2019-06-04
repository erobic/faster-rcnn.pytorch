# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import numpy as np
import argparse
import pprint
import time
import cv2
import torch
from torch.autograd import Variable

import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread
from lib.roi_data_layer.roidb import combined_roidb
from lib.roi_data_layer.roibatchLoader import roibatchLoader
from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from lib.model.rpn.bbox_transform import clip_boxes
from lib.model.nms.nms_wrapper import nms
from lib.model.rpn.bbox_transform import bbox_transform_inv
from lib.model.utils.blob import im_list_to_blob
from lib.model.faster_rcnn.vgg16 import vgg16
from lib.model.faster_rcnn.resnet import resnet
import pdb
import json
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt

# format: xmin, ymin, xmax, ymax
try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

num_fixed_boxes = 15
feature_length = 2048


def id_to_clevr_filename(image_id, split, extension='.png'):
    return 'CLEVR_' + split + '_' + str(image_id).rjust(6, '0') + extension


def id_to_coco_filename(id, split, extension='.jpg'):
    if split == 'train' or split == 'val':
        year = '2014'
    else:
        year = '2015'
    return 'COCO_' + split + year + "_" + str(id).rjust(12, '0') + extension


def coco_filename_to_id(filename):
    return int(filename.split("_")[2].split(".")[0])


def clevr_filename_to_id(filename):
    return int(filename.split("_")[2].split(".")[0])


def filename_to_id(filename):
    if 'clevr' in filename.lower():
        return clevr_filename_to_id(filename)
    else:
        return coco_filename_to_id(filename)


def id_to_filename(image_id, split, dataset):
    if dataset.lower().startwith('clevr'):
        return id_to_clevr_filename(image_id, split)
    else:
        return id_to_coco_filename(id, split)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/res101.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models',
                        default="/hdd/robik/FasterRCNN/models")
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=10021, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    parser.add_argument('--webcam_num', dest='webcam_num',
                        help='webcam ID number',
                        default=-1, type=int)

    parser.add_argument('--root', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--image_dir', required=False, default=None)
    parser.add_argument('--image_limit', required=False, default=None, type=int)
    parser.add_argument('--visualize_only', action='store_true')
    parser.add_argument('--use_oracle_gt_boxes', action='store_true')
    parser.add_argument('--num_images', default=None, type=int)
    parser.add_argument('--visualize_subdir', default='visualize_faster_rcnn')
    parser.add_argument('--load_subdir', required=False)

    args = parser.parse_args()
    args.dataroot = args.root + '/' + args.dataset
    if args.image_dir is None:
        args.image_dir = args.dataroot + '/images/' + args.split
    print("image_dir: {}".format(args.image_dir))
    args.visualize_dir = args.dataroot + '/' + args.visualize_subdir
    if not os.path.exists(args.visualize_dir):
        os.mkdir(args.visualize_dir)

    args.scenes_filepath = os.path.join(args.dataroot, 'faster-rcnn', '{}_scenes_with_bb.json'.format(args.split))
    if os.path.exists(args.scenes_filepath):
        with open(args.scenes_filepath) as scenes_file:
            args.scenes = json.load(scenes_file)
    if args.load_subdir is None:
        args.load_subdir = args.dataset
    return args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def draw_preds(im2show, boxes, classes, score_class_ixs, scores):
    for ix, class_ix in enumerate(score_class_ixs):
        curr_box = boxes[ix]
        bbox = (int(curr_box[0]), int(curr_box[1]), int(curr_box[2]), int(curr_box[3]))
        cv2.rectangle(im2show, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 204, 0), 2)
        class_name = classes[class_ix]
        cv2.putText(im2show, '%s: %.3f' % (class_name, scores[ix][class_ix]), (bbox[0], bbox[1] + 15),
                    cv2.FONT_HERSHEY_PLAIN,
                    0.6, (0, 0, 255), thickness=1)
    return im2show


def extract_imglist_from_scenes(scenes, num_images=None):
    image_ids = []
    image_files = []
    counter = 0
    for ann in scenes['annotations']:
        if num_images is not None and counter > num_images:
            break
        image_ids.append(ann['image_id'])
        image_files.append(ann['filename'])
        counter += 1
    return image_ids, image_files


def extract_imglist(fn_list, num_images=None):
    image_ids, image_files = [], []
    counter = 0
    for fn in fn_list:
        if num_images is not None and counter > num_images:
            break
        image_id = filename_to_id(fn)
        image_ids.append(image_id)
        image_files.append(fn)
        counter += 1
    return image_ids, image_files


def extract_gt_rois(objects):
    rois = []
    for ix in range(num_fixed_boxes):
        if ix < len(objects):
            obj = objects[ix]
            # print("obj[xm,ax]: {}".format(obj['xmax']))
            roi = [0, obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']]
        else:
            # pad with global context
            roi = [0, 0, 0, 480, 320]  # TODO: Do not use fixed dims for other datasets
        rois.append(roi)
    rois = np.array(rois).astype(np.float32)
    return rois


if __name__ == '__main__':
    printed = False
    args = parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = args.cuda

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.

    input_dir = args.load_dir + "/" + args.net + "/" + args.load_subdir.lower()

    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
                             'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    with open(args.dataroot + '/faster-rcnn/objects_count.json') as ovf:
        classes = list(json.load(ovf).keys())

    # initialize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    if args.cuda > 0:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    # initialize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda > 0:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data, volatile=True)
    im_info = Variable(im_info, volatile=True)
    num_boxes = Variable(num_boxes, volatile=True)
    gt_boxes = Variable(gt_boxes, volatile=True)

    if args.cuda > 0:
        cfg.CUDA = True

    if args.cuda > 0:
        fasterRCNN.cuda()

    fasterRCNN.eval()

    start = time.time()
    max_per_image = 100
    vis = True

    imglist = sorted(os.listdir(args.image_dir))
    num_images = len(imglist)

    if args.image_limit is not None:
        imglist = imglist[0:args.image_limit]
        num_images = len(imglist)
        print("num_images {}".format(num_images))

    image_ids, image_files = extract_imglist(imglist, args.num_images)
    num_images = len(image_ids)

    print('Loaded Photo: {} images.'.format(num_images))

    ### Init h5 file
    if not args.visualize_only:
        if args.use_oracle_gt_boxes:
            feat_dir = 'oracle-features'
        else:
            feat_dir = 'features'

        h5_filename = args.dataroot + '/{}/{}.hdf5'.format(feat_dir, args.split)
        h5_file = h5py.File(h5_filename, "w")
        h5_img_features = h5_file.create_dataset(
            'image_features', (num_images, num_fixed_boxes, feature_length), 'f')
        h5_spatial_img_features = h5_file.create_dataset(
            'spatial_features', (num_images, num_fixed_boxes, 6), 'f')
        indices = {'image_id_to_ix': {}, 'image_ix_to_id': {}}

    counter = 0
    print("num_images: {}".format(num_images))

    for image_ix in tqdm(iter(range(num_images))):
        total_tic = time.time()

        im_file = os.path.join(args.image_dir, image_files[image_ix])
        img_id = image_ids[image_ix]
        if not printed:
            print("im_id: {}".format(img_id))

        im_in = np.array(imread(im_file, mode='RGB'))
        height, width = im_in.shape[0], im_in.shape[1]
        if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)
        # rgb -> bgr
        im = im_in[:, :, ::-1]

        blobs, im_scales = _get_image_blob(im)
        im_blob = blobs
        im_data_pt = torch.from_numpy(im_blob)
        im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
        det_tic = time.time()
        im_data = im_data.permute(0, 3, 1, 2)
        feats = fasterRCNN.extract_base_feat(im_data)
