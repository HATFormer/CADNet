import os
import cv2
from tqdm import tqdm
import numpy as np

def find_type(path):
    if 'line' in path:
        TYPE = 'line'
    elif 'abpt' in path:
        TYPE = 'abpt'
    else:
        TYPE = 'mixed_hard'
    return TYPE

def find_gt(img_path, outpath):
    root = r'/home/hc/lby/ccd/mmsegmentation/data/SynLCD/Possion'
    name = img_path.split('/')[-1].split('_')[-1]
    bg = [_ for _ in img_path.split('/') if 'bg' in _][-1].split('_')[0]
    if int(name.strip('.png'))<=300:
        gt_path = os.path.join(root, bg, find_type(img_path), 'gt', name)
        gt = cv2.imread(gt_path)
    else:
        gt = np.zeros((512,512),dtype=np.uint8)
    line_img = np.zeros_like(gt)
    line_img[gt==1] = 255
    abpt_img = np.zeros_like(gt)
    abpt_img[gt==2] = 255

    try:
        cv2.imwrite(os.path.join(outpath,'line',img_path.split('/')[-1]),line_img)
        cv2.imwrite(os.path.join(outpath,'abpt',img_path.split('/')[-1]),abpt_img)
    except:
        print(1)

if __name__ == '__main__':
    inpath = r'/home/hc/lby/ccd/mmsegmentation/work_dirs/FCN_synLCD/eval_output/abpt'
    outpath = r'/home/hc/lby/ccd/mmsegmentation/work_dirs/mixed_gt/'
    img_list = os.listdir(inpath)
    for img_path in tqdm(img_list):
        find_gt(img_path, outpath)
