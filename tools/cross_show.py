import cv2
import random
import matplotlib.pyplot as plt
import numpy as np
import os


def extract_list(root):
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
    if 'line' in path:
        TYPE = 'line'
    elif 'abpt' in path:
        TYPE = 'abpt'
    else:
        TYPE = 'mixed_hard'
    return TYPE
def extract_img(name_list, num=3):
    img_list = []
    keys = list(name_list[0].keys())
    # TODO: automatically select the results where our method is at the first place
    best_list = []
    for midx, method in enumerate(name_list):
        best = [[] for _ in range(num)]
        bgs = list(method.keys())
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
    for midx,method in enumerate(name_list):
        mtmp = []
        for pidx in range(num):
            bg_id = keys[pidx // (num // len(keys))]
            pop_id = best_list[midx][pidx][0]
            img_path = name_list[midx][bg_id]['ng'][pop_id]
            TYPE = find_type(img_path)
            ok, ng, gt = find_corr_img(bg_id, img_path, TYPE)
            tmp = [ok, ng, gt]
            #img_path = method[bg_id]['ng'].pop(pop_id)
            tmp.append(cv2.imread(img_path))
            mtmp.append(tmp)
        img_list.append(mtmp)
    return img_list


def show1(img_list, name=[], save_dir='cross_test.png'):
    rows = len(img_list)
    cols = len(img_list[0])
    plt.figure()
    plt.clf()
    # f, ax = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    plt.tight_layout()
    # f.tight_layout()
    for row_id in range(1, rows + 1):
        for col_id in range(1, cols + 1):
            # ax[idx, 2].imshow(toArray(gt[idx]), cmap='gray')
            # ax[idx, 2].set_xticks([])
            # ax[idx, 2].set_yticks([])
            plt.subplot(rows, cols, (row_id - 1) * cols + col_id)
            plt.xticks([])
            plt.yticks([])
            image = img_list[row_id - 1][col_id - 1][:, :, ::-1]
            plt.imshow(image)

    plt.savefig(save_dir, dpi=300, pad_inches=0.1, bbox_inches='tight', )
    plt.close()


def show(img_list, names=[], save_dir='test.png'):
    fea1 = cv2.imread(r'work_dirs/siameseCGNet_pretrained111/mixed/fea_vis1/feamap0/bg10_103_feavis0.png')
    fea2 = cv2.imread(r'work_dirs/siameseCGNet_pretrained111/mixed/fea_vis1/feamap1/bg10_103_feavis1.png')
    tsne = cv2.imread(r'work_dirs/siameseCGNet_pretrained111/mixed/fea_vis1/tsne1/tsne_vis_5.png')
    tsne = cv2.resize(tsne,(512,512))
    plus_imgs = [fea1,fea2,tsne]
    rows = len(img_list[0])
    cols = len(img_list[0][0])
    f, ax = plt.subplots(rows*2, cols+3, figsize=(cols * 1.5, rows * 1.8))
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
            for plus in range(1,4):
                ax[row_id+rows*i, col_id+plus].imshow(cv2.cvtColor(plus_imgs[plus-1],cv2.COLOR_BGR2RGB))
                ax[row_id+rows*i, col_id+plus].set_xticks([])
                ax[row_id+rows*i, col_id+plus].set_yticks([])
    for _ in range(ax.shape[1]):
        ax[row_id+rows*i, _].set_xlabel(names[_], fontsize=8)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(save_dir, dpi=300, pad_inches=0.05, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    roots = [r'work_dirs/siameseCGNet_pretrained111/line_v9/output/abpt',
             r'work_dirs/siameseCGNet_pretrained111/line_v9/output/line',
             r'work_dirs/siameseCGNet_pretrained111/abpt_v4/output/abpt',
             r'work_dirs/siameseCGNet_pretrained111/abpt_v4/output/line']
    name_list = [extract_list(os.path.join(root, '')) for root in roots]
    img_list = extract_img(name_list)
    show(img_list, names=['OK', 'NG', 'GT', 'Pred','mid-feamap','out-feamap','tsne'])
    print(1)
