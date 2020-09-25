import cv2
import numpy as np
import json
mask_definition_json = json.load(open('data/json/annotation_definitions.json', 'r'))
mask_definitions = {}
for spec in mask_definition_json['annotation_definitions'][1]['spec']:
    label_name = spec['label_name']
    r = round(spec['pixel_value']['r'] * 255)
    g = round(spec['pixel_value']['g'] * 255)
    b = round(spec['pixel_value']['b'] * 255)
    rgb = np.array([b, g, r], dtype=np.int64)
    mask_definitions[label_name] = rgb


im = cv2.imread('data/mask/segmentation_1.png')
im = cv2.resize(im, (500,600), interpolation=cv2.INTER_NEAREST)
# im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

base = None
import time
s0 = time.time()
for key, value in mask_definitions.items():
    mask = np.zeros(shape=(im.shape[:2]))
    mask[np.all(np.abs(im - value) < 2, axis=-1)] = 255
    mask = np.expand_dims(mask, axis=0)
    if base is None:
        base = mask
    else:
        base = np.concatenate((base, mask), axis=0)

sample_im = cv2.imread('data/original/rgb_1.png')
sample_im = cv2.resize(sample_im, (500, 600), interpolation=cv2.INTER_NEAREST)

original = np.zeros((600, 500, 3), dtype=np.uint8)


for mask, label_name in zip(base, mask_definitions.keys()):
    if label_name in ['Card_Background', 'KOR', 'Person_Image']:
        rgb = mask_definitions[label_name]
        # rgba = np.append(rgb, [10])
        indices = mask > 25
        if label_name in ['KOR']:
            original[indices] = (0, 255, 0)
        elif label_name in ['Person_Image']:
            original[indices] = (0, 0, 255)

cv2.imshow('1', sample_im)
# dst = cv2.addWeighted(sample_im, 0.7, original, 0.3, 0)

mask_indices = np.any(original > 0, axis=-1)
sample_im[mask_indices] = sample_im[mask_indices] * 0.5 + original[mask_indices] * 0.5

cv2.imshow('base', sample_im)
print(time.time() - s0)
# for index, data in enumerate(base):
#     cv2.imshow(f'{index}', data)

cv2.waitKey(0)