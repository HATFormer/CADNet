import cv2
import os
from tqdm import tqdm
import numpy as np



def pad(root,bg, type):
    W = 1280
    H = 768
    root += '/{}'.format(bg)
    path = os.path.join(root,type,'defect')
    names = os.listdir(path)
    out_path = os.path.join(r'/media/hc/KINGSTON',bg, type)
    os.makedirs(out_path,exist_ok=True)
    print('Processing {}...'.format(path))
    for name in tqdm(names):
        img = cv2.imread(os.path.join(path,name))
        h,w,c = img.shape
        out = np.zeros((H,W,c), dtype=img.dtype)
        x,y = int(W/2-w/2), int(H/2-h/2)
        out[y:y+h,x:x+w,:] = img
        cv2.imwrite(os.path.join(out_path,name),out)

if __name__ == '__main__':
    root = r'/home/hc/lby/ccd/mmsegmentation/data/SynLCD/Possion'
    type = ['abpt', 'line', 'mixed_hard']
    for i in range(2,11):
        for j in range(3):
            pad(root,'bg{}'.format(i), type[j])