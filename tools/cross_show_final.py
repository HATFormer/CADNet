import cv2
import random
import matplotlib.pyplot as plt
import numpy as np
import os


def extract_list(root):
    print('Extracting image list...')
    img_dict = {'bg8': {'ok': [], 'ng': []},
                'bg9': {'ok': [], 'ng': []},
                'bg10': {'ok': [], 'ng': []}}
    names = os.listdir(root)
    for name in names:
        pre, idx = name.split('_')
        if int(idx.strip('.png')) > 300:
            img_dict[pre]['ok'].append(os.path.join(root, name))
        else:
            img_dict[pre]['ng'].append(os.path.join(root, name))
    return img_dict


def find_corr_img(bg_id, img_path, TYPE='mixed_hard', only_gt=False):
    root = r'/home/hc/lby/ccd/mmsegmentation/data/SynLCD/Possion'
    name = img_path.split('/')[-1].split('_')[-1]
    gt = cv2.imread(os.path.join(root, bg_id, TYPE, 'gt', name))
    gt[gt != 0] = 255
    if only_gt:
        return gt
    ok = cv2.imread(os.path.join(root, bg_id.lstrip('bg') + '.png'))
    ng = cv2.imread(os.path.join(root, bg_id, TYPE, 'defect', name))
    return ok, ng, gt

def find_type(path):
    if 'line' in path.split('/')[-3]:
        TYPE = 'line'
    elif 'abpt' in path.split('/')[-3]:
        TYPE = 'abpt'
    else:
        TYPE = 'mixed_hard'
    return TYPE
def extract_img(name_list, num=2):
    print('Extracting image...')
    img_list = []
    keys = list(name_list[0].keys())[:2]
    # TODO: automatically select the results where our method is at the first place
    best_list = []
    for midx, method in enumerate([name_list[0],name_list[2]]):
        best = [[] for _ in range(num)]
        bgs = list(method.keys())[:2]
        for bid, bg in enumerate(bgs):
            min_err = 1e+6
            for pid, img_path in enumerate(method[bg]['ng']):
                TYPE = find_type(img_path)
                gt = find_corr_img(bg, img_path, TYPE, only_gt=True)
                tmp = gt-cv2.imread(img_path)
                errs = np.sum(tmp==255)
                if errs<min_err:
                    min_err = errs
                    best[bid] = [pid, min_err]
        best_list.append(best)
    for midx,method in enumerate([name_list[0],name_list[2]]):
        mtmp = []
        for pidx in range(num):
            bg_id = keys[pidx // (num // len(keys))]
            pop_id = best_list[midx][pidx][0]
            img_path = method[bg_id]['ng'][pop_id]
            TYPE = find_type(img_path)
            plus_path = name_list[3 if midx==0 else 1][bg_id]['ng'][pop_id]
            ok, ng, gt = find_corr_img(bg_id, img_path, TYPE)
            tmp = [ok, ng, gt]
            #img_path = method[bg_id]['ng'].pop(pop_id)
            tmp.append(cv2.imread(plus_path))
            tmp.append(cv2.imread(img_path))
            mtmp.append(tmp)
        img_list.append(mtmp)
    return img_list


def show(img_list, names=[], save_dir='cross_test.png'):
    print('Showing image...')
    rows = len(img_list[0])
    cols = len(img_list[0][0])
    f, ax = plt.subplots(rows*2, cols, figsize=(cols * 1.5, rows * 3.0))
    plt.tight_layout()
    plt.subplots_adjust(wspace=1,hspace=1)
    f.tight_layout()
    for i,sub_list in enumerate(img_list):
        for row_id in range(rows):
            for col_id in range(cols):
                image = sub_list[row_id][col_id][:, :, ::-1]
                ax[row_id+rows*i, col_id].imshow(image)
                ax[row_id+rows*i, col_id].set_xticks([])
                ax[row_id+rows*i, col_id].set_yticks([])

    for _ in range(ax.shape[1]):
        ax[row_id+rows*i, _].set_xlabel(names[_], fontsize=12)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(save_dir, dpi=300, pad_inches=0.05, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    roots = [r'work_dirs/siameseCGNet_pretrained111/line_v9/output/abpt',
             r'work_dirs/siameseCGNet_pretrained111/line_v9/output/line',
             r'work_dirs/siameseCGNet_pretrained111/abpt_v4/output/line',
             r'work_dirs/siameseCGNet_pretrained111/abpt_v4/output/abpt']
    name_list = [extract_list(os.path.join(root, 'mixed_output')) for root in roots]
    img_list = extract_img(name_list)
    show(img_list, names=['OK', 'NG', 'GT', 'ICPred','OOCPred'])
    print(1)
