import cv2
import json
import numpy as np
from collections import OrderedDict


def load_segmentation_definition(json_path):
    mask_definition_json = json.load(open(json_path, 'r'))
    mask_definitions = {}
    for spec in mask_definition_json['annotation_definitions'][1]['spec']:
        label_name = spec['label_name']
        r = round(spec['pixel_value']['r'] * 255)
        g = round(spec['pixel_value']['g'] * 255)
        b = round(spec['pixel_value']['b'] * 255)
        rgb = np.array([b, g, r], dtype=np.int64)
        mask_definitions[label_name] = rgb
    return mask_definitions


def segmentation_to_masks(mask_definitions, gt_path, size=(500, 600), show=False):
    gt_im = cv2.imread(gt_path)
    gt_im = cv2.resize(gt_im, size, cv2.INTER_NEAREST)
    masks = None
    for key, value in mask_definitions.items():
        mask = np.zeros(shape=(gt_im.shape[:2]))
        mask[np.all(np.abs(gt_im - value) < 2, axis=-1)] = 255
        mask = np.expand_dims(mask, axis=0)
        if masks is None:
            masks = mask
        else:
            masks = np.concatenate((masks, mask), axis=0)

    if show:
        for index, mask in enumerate(masks):
            cv2.imshow(f'mask-{index}', mask)
        cv2.waitKey(0)
    return masks


def masks_to_segmentation(mask_definitions, masks, threshold=0.1, show=False):
    (channel, height, width) = np.shape(masks)
    segmentation_im = np.zeros((height, width, 3), dtype=np.uint8)
    for mask, label_name in zip(masks[1:], mask_definitions.keys()):
        if label_name in ['Card_Background', 'KOR', 'Person_Image']:
            rgb = mask_definitions[label_name]
            indices = mask > threshold
            if label_name in ['KOR']:
                segmentation_im[indices] = rgb
            elif label_name in ['Person_Image']:
                segmentation_im[indices] = rgb
    if show:
        cv2.imshow('segmentation', segmentation_im)
        cv2.waitKey(0)
    return segmentation_im


def overaly_segmentation(sample_im, mask, alpha=0.5, show=False):
    mask_indices = np.any(mask > 0, axis=-1)
    sample_im[mask_indices] = sample_im[mask_indices] * alpha + mask[mask_indices] * alpha
    if show:
        cv2.imshow('base', sample_im)
        cv2.waitKey(0)
    return sample_im


def copy_state_dict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict



if __name__ == '__main__':
    json_path = 'data/json/annotation_definitions.json'
    definition = load_segmentation_definition(json_path)
    gt_masks = segmentation_to_masks(definition, 'data/mask/segmentation_1.png', show=True)

    im = cv2.imread('data/original/rgb_1.png')
    im = cv2.resize(im, (500, 600), interpolation=cv2.INTER_NEAREST)
    segmentation = masks_to_segmentation(definition, gt_masks, show=True)
    overaly_segmentation(im, segmentation, alpha=0.7, show=True)
