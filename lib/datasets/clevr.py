from __future__ import print_function
from __future__ import absolute_import
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import gzip
import PIL
import json
from .vg_eval import vg_eval
from model.utils.config import cfg
import pickle
import pdb

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

ROOT = '/hdd/robik'
DATASET = 'CLEVR'
DATA_ROOT = os.path.join(ROOT, DATASET)
FASTER_RCNN_ROOT = os.path.join(DATA_ROOT, 'faster-rcnn')
split = 'train'


class clevr(imdb):
    def __init__(self, image_set, ):
        print("image_set: {}".format(image_set))
        # raise ValueError
        imdb.__init__(self, 'clevr_' + image_set)
        self._image_set = image_set
        self._data_path = FASTER_RCNN_ROOT
        self._img_path = os.path.join(DATA_ROOT, 'faster-rcnn', 'xml')

        # Load classes
        self._classes = []
        self._class_to_ind = {}
        self._image_index = []
        with open(os.path.join(self._data_path, '{}_scenes_with_bb.json'.format(split))) as f:
            self.scenes_with_bb = json.load(f)
            label_to_ix = self.scenes_with_bb['label_to_ix']
            ix_to_label = self.scenes_with_bb['ix_to_label']
            self._class_to_ind = ix_to_label
            self._classes = label_to_ix.keys()
            self.annotations = self.scenes_with_bb['annotations']
            for ann in self.annotations:
                self._image_index.append(ann['image_id'])

        self._image_ext = '.png'
        self._roidb_handler = self.gt_roidb

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i
        # return self._image_index[i]

    def clevr_index_to_filename(self, index, split, ext='.png'):
        if ext is None:
            ext = '.png'
        return 'CLEVR_' + split + '_' + str(index).rjust(6, '0') + ext

    def image_path_from_index(self, index):
        return os.path.join(DATA_ROOT, 'images', self._image_set, self.clevr_index_to_filename(index, self._image_set))

    @property
    def cache_path(self):
        cache_path = os.path.join(FASTER_RCNN_ROOT, 'cache')
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            fid = gzip.open(cache_file, 'rb')
            roidb = pickle.load(fid)
            fid.close()
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_clevr_annotation(index)
                    for index in self.image_index]
        fid = gzip.open(cache_file, 'wb')
        pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        fid.close()
        print('wrote gt roidb to {}'.format(cache_file))
        return gt_roidb

    def _get_size(self, index):
        return PIL.Image.open(self.image_path_from_index(index)).size

    def _annotation_path(self, index):
        return os.path.join(self._data_path, 'xml', self._image_set,
                            self.clevr_index_to_filename(index, self._image_set, '.xml'))

    def _load_clevr_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        width, height = self._get_size(index)
        filename = self._annotation_path(index)
        tree = ET.parse(filename)
        objs = tree.findall('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        obj_dict = {}
        ix = 0
        for obj in objs:
            obj_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            cls = self._class_to_ind[obj_name]
            obj_dict[obj.find('object_id').text] = ix
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
            ix += 1
        # clip gt_classes and gt_relations
        #gt_classes = gt_classes[:ix] # WTF?
        overlaps = scipy.sparse.csr_matrix(overlaps)

        # gt_relations, gt_attributes
        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'width': width,
                'height': height,
                'flipped': False,
                'seg_areas': seg_areas}

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(self.classes, all_boxes, output_dir)
        self._do_python_eval(output_dir)

    def _get_vg_results_file_template(self, output_dir):
        filename = 'detections_' + self._image_set + '_{:s}.txt'
        path = os.path.join(output_dir, filename)
        return path

    def _write_voc_results_file(self, classes, all_boxes, output_dir):
        for cls_ind, cls in enumerate(classes):
            if cls == '__background__':
                continue
            print('Writing "{}" CLEVR results file'.format(cls))
            filename = self._get_vg_results_file_template(output_dir).format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self._image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(str(index), dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir, pickle=True):
        # We re-use parts of the pascal voc python code for visual genome
        aps = []
        nposs = []
        thresh = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        # Load ground truth
        gt_roidb = self.gt_roidb()
        classes = self._classes
        for i, cls in enumerate(classes):
            filename = self._get_vg_results_file_template(output_dir).format(cls)
            rec, prec, ap, scores, npos = vg_eval(
                filename, gt_roidb, self._image_index, i, ovthresh=0.5,
                use_07_metric=use_07_metric, eval_attributes=False)

            # Determine per class detection thresholds that maximise f score
            if npos > 1:
                f = np.nan_to_num((prec * rec) / (prec + rec))
                thresh += [scores[np.argmax(f)]]
            else:
                thresh += [0]
            aps += [ap]
            nposs += [float(npos)]
            print('AP for {} = {:.4f} (npos={:,})'.format(cls, ap, npos))
            if pickle:
                with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                    pickle.dump({'rec': rec, 'prec': prec, 'ap': ap,
                                 'scores': scores, 'npos': npos}, f)

        # Set thresh to mean for classes with poor results
        thresh = np.array(thresh)
        avg_thresh = np.mean(thresh[thresh != 0])
        thresh[thresh == 0] = avg_thresh
        filename = 'object_thresholds_' + self._image_set + '.txt'
        path = os.path.join(output_dir, filename)
        with open(path, 'wt') as f:
            for i, cls in enumerate(classes[1:]):
                f.write('{:s} {:.3f}\n'.format(cls, thresh[i]))

        weights = np.array(nposs)
        weights /= weights.sum()
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('Weighted Mean AP = {:.4f}'.format(np.average(aps, weights=weights)))
        print('Mean Detection Threshold = {:.3f}'.format(avg_thresh))
        print('~~~~~~~~')
        print('Results:')
        for ap, npos in zip(aps, nposs):
            print('{:.3f}\t{:.3f}'.format(ap, npos))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** PASCAL VOC Python eval code.')
        print('--------------------------------------------------------------')


if __name__ == '__main__':
    d = clevr('train')
    res = d.roidb
    from IPython import embed;
    embed()
