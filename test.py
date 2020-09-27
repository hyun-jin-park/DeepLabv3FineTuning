import os
import cv2
import time
import glob
import torch
import argparse

from model import createDeepLabv3
from utils import masks_to_segmentation, overaly_segmentation, load_segmentation_definition, copy_state_dict

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path", default='./out/weights.pt', type=str)
parser.add_argument(
    "--data_directory", default='./data/original/', type=str)
parser.add_argument(
    "--result_directory", default='./result/', type=str)

args = parser.parse_args()
if not os.path.isdir(args.result_directory):
    os.mkdir(args.result_directory)

model = createDeepLabv3(outputchannels=12, pretrain=True)
# model = torch.nn.DataParallel(model)
train_model = torch.load(args.model_path)
trained_state_dict = copy_state_dict(train_model.state_dict())
model.load_state_dict(trained_state_dict)
model.eval()
model.cuda()

definitions = load_segmentation_definition('data/json/annotation_definitions.json')
s0 = time.time()
with torch.no_grad():
    for step, path in enumerate(glob.glob(args.data_directory + '*.png')):
        im = cv2.imread(path)
        im = cv2.resize(im, (480, 320), cv2.INTER_NEAREST)
        input_tensor = torch.from_numpy(im)
        input_tensor = input_tensor.type(torch.FloatTensor) / 255
        input_tensor = input_tensor.cuda()
        input_tensor = input_tensor.permute(2, 0, 1)
        input_tensor = torch.unsqueeze(input_tensor, 0)
        output = model(input_tensor)['out'][0]
        masks = output.data.cpu().numpy()
        # masks = output.argmax(0)
        segmentation = masks_to_segmentation(definitions, masks, threshold=0.7)
        overlay_image = overaly_segmentation(im, segmentation, show=True)
        cv2.imwrite(path.replace('original', 'overlay'), overlay_image)
        if step > 10:
            break
print(f'elapsed time: {time.time() - s0}')



