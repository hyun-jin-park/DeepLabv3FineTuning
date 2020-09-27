import os
import glob
import shutil
import random
import tqdm


def get_child_folder_name(parent_path, post_fix):
    folder_list = glob.glob(parent_path + f'/{post_fix}*')
    if len(folder_list) > 1:
        raise Exception(f'There are many {post_fix} Folder')
    elif len(folder_list) == 0:
        raise Exception(f'There is no {post_fix} Folder')
    return folder_list[0]


if __name__ == '__main__':
    src_path = '/home/embian/Unity_Dataset/2020_09_25/'
    target_folder = '/home/hjpark/OCR/DeepLabv3FineTuning/data/'

    for index, id_type in tqdm.tqdm(enumerate(os.listdir(src_path))):
        print(id_type)
        data_path = os.path.join(src_path, id_type)
        rgb_folder = get_child_folder_name(data_path, 'RGB')
        mask_folder = get_child_folder_name(data_path, 'SemanticSegmentation')
        for path in glob.glob(rgb_folder + '/*.png'):
            base_name = os.path.basename(path)

            if random.randint(1, 10) > 8:
                train_eval = 'Test'
            else:
                train_eval = 'Train'

            target_rgb_path = os.path.join(target_folder, train_eval, 'original', f'{id_type}_{base_name}')

            mask_file_name = base_name.replace('rgb_', 'segmentation_')
            src_mask_path =  os.path.join(mask_folder, mask_file_name)
            target_mask_path = os.path.join(target_folder, train_eval, 'mask', f'{id_type}_{mask_file_name}')

            shutil.move(path, target_rgb_path)
            shutil.move(src_mask_path, target_mask_path)
            # print(f'{src_mask_path} --> {target_mask_path}')
            # print(f'{path} --> {target_rgb_path}')
