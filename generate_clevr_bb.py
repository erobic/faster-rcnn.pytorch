'''
https://github.com/larchen/clevr-vqa/blob/master/bounding_box.py
'''

'''
   Copyright 2017 Larry Chen

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''

import argparse
import json
import os

import cv2

WIDTH = 480
HEIGHT = 320


def generate_label_map():
    sizes = ['large', 'small']
    colors = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
    materials = ['rubber', 'metal']
    shapes = ['cube', 'sphere', 'cylinder']

    names = [s + ' ' + c + ' ' + m + ' ' + sh for s in sizes for c in colors for m in materials for sh in shapes]

    with open(os.path.join(SCENES_DIR, 'clevr_label_map.txt'), 'w') as f:
        [f.write('item {\n  id: %d\n  name: \'%s\'\n}\n\n' % (i + 1, name)) for i, name in enumerate(names)]
        f.close()

    return names


def scene_to_annotation(scene, names, printed, object_id):
    objs = scene['objects']
    rotation = scene['directions']['right']
    annotations = []
    for i, obj in enumerate(objs):
        ann = {}
        [x, y, z] = obj['pixel_coords']
        [x1, y1, z1] = obj['3d_coords']

        cos_theta, sin_theta, _ = rotation
        if not printed:
            print("x1: {} y1: {} cos_theta: {}, sin_theta: {}".format(x1, y1, cos_theta, sin_theta))

        x1 = x1 * cos_theta + y1 * sin_theta
        y1 = x1 * -sin_theta + y1 * cos_theta
        if not printed:
            print("After rotation: x1: {} y1: {}".format(x1, y1))

        height_d = 6.9 * z1 * (15 - y1) / 2.0  # erobic: Not sure where these numbers come from
        height_u = height_d
        width_l = height_d
        width_r = height_d

        if obj['shape'] == 'cylinder':
            d = 9.4 + y1
            h = 6.4
            s = z1

            height_u *= (s * (h / d + 1)) / ((s * (h / d + 1)) - (s * (h - s) / d))
            height_d = height_u * (h - s + d) / (h + s + d)

            width_l *= 11 / (10 + y1)
            width_r = width_l

        if obj['shape'] == 'cube':
            height_u *= 1.3 * 10 / (10 + y1)
            height_d = height_u
            width_l = height_u
            width_r = height_u

        obj_name = obj['size'] + ' ' + obj['color'] + ' ' + obj['material'] + ' ' + obj['shape']

        ymin = max(0, (y - height_d) / 320.0)
        ymax = min(1, (y + height_u) / 320.0)
        xmin = max(0, (x - width_l) / 480.0)
        xmax = min(1, (x + width_r) / 480.0)

        ann = {
            'xmin': int(xmin * (WIDTH - 1)),
            'ymin': int(ymin * (HEIGHT - 1)),
            'xmax': int(xmax * (WIDTH - 1)),
            'ymax': int(ymax * (HEIGHT - 1)),
            'label_id': names.index(obj_name),
            'label': obj_name,
            'object_id': object_id,
            'size': obj['size'],
            'color': obj['color'],
            'material': obj['material'],
            'shape': obj['shape']
        }
        object_id += 1
        annotations.append(ann)
    return annotations, object_id


def convert_clevr_scene(scene_file, names):
    with open(scene_file) as sf:
        scene_data = json.load(sf)
        scenes = scene_data['scenes']

    printed = False
    annotations = []
    annotation_holder = {'label_to_ix': {}, 'ix_to_label': {}}
    object_id = 0
    for scene in scenes:
        objects, object_id = scene_to_annotation(scene, names, printed, object_id)
        ann = {
            'filename': str(scene['image_filename']),
            'image_id': scene['image_index'],
            'objects': objects
        }

        annotations.append(ann)
        if not printed:
            print("Annotation: {}".format(ann))
        printed = True
    annotation_holder['annotations'] = annotations

    for label_ix, label in enumerate(names):
        annotation_holder['label_to_ix'][label_ix] = label
        annotation_holder['ix_to_label'][label] = label_ix
    print(annotation_holder['label_to_ix'])
    print(annotation_holder['ix_to_label'])
    return annotation_holder


def save_bb(names, scene_file, out_bb_file):
    converted = convert_clevr_scene(scene_file, names)
    with open(out_bb_file, 'w') as f:
        json.dump(converted, f)


def render_few_examples(bb_file, image_dir, out_dir):
    height = 320
    width = 480

    with open(bb_file) as f:
        anns = json.load(f)['annotations']
    total_examples = 0
    for ann in anns:
        img_filename = os.path.join(image_dir, ann['filename'])
        img = cv2.imread(img_filename)
        for obj in ann['objects']:
            cv2.rectangle(img, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color=(0, 255, 0), thickness=2)
        out_file = os.path.join(out_dir, ann['filename'])
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        cv2.imwrite(out_file, img)
        print("Saved: {}".format(out_file))
        # cv2.destroyAllWindows()

        total_examples += 1
        if total_examples > 100:
            return


def main(has_train=True):
    names = generate_label_map()
    if has_train:
        save_bb(names, os.path.join(SCENES_DIR, 'CLEVR_train_scenes.json'),
                os.path.join(FASTER_RCNN_DIR, 'train_scenes_with_bb.json'))
    save_bb(names, os.path.join(SCENES_DIR, 'CLEVR_val_scenes.json'),
            os.path.join(FASTER_RCNN_DIR, 'val_scenes_with_bb.json'))
    render_few_examples(os.path.join(FASTER_RCNN_DIR, 'train_scenes_with_bb.json'),
                        os.path.join(DATA_ROOT, 'images', 'train'),
                        os.path.join(DATA_ROOT, 'sample_bbs'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--has_train', action='store_false')
    args = parser.parse_args()
    DATA_ROOT = '/hdd/robik/'+args.dataset
    SCENES_DIR = DATA_ROOT + '/scenes'
    FASTER_RCNN_DIR = DATA_ROOT + '/faster-rcnn'

    main()
