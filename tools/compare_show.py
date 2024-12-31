import cv2
import random
import matplotlib.pyplot as plt
import numpy as np
import os


def extract_list(root):
    img_dict = {'bg8': {'ok': [], 'ng': []},
                'bg9': {'ok': [], 'ng': []},
                'bg10': {'ok': [], 'ng': []}}
    base = '/home/hc/lby/ccd/mmsegmentation/work_dirs/segformer/eval_output/mixed_output'
    names = os.listdir(root)
    names = [x for x in names if x in os.listdir(base)]
    for name in names:
        pre, idx = name.split('_')
        if int(idx.strip('.png')) > 300:
            img_dict[pre]['ok'].append(os.path.join(root, name))
        else:
            img_dict[pre]['ng'].append(os.path.join(root, name))
    return img_dict

def find_corr_img(bg_id, img_path, TYPE='mixed_hard', only_gt=False):
    root = r'/home/hc/lby/ccd/mmsegmentation/data/SynLCD/'
    name = img_path.split('/')[-1].split('_')[-1]
    gt = cv2.imread(os.path.join(root, bg_id, TYPE, 'gt', name))
    gt[gt != 0] = 255
    if only_gt:
        return gt
    ok = cv2.imread(os.path.join(root, bg_id.lstrip('bg') + '.png'))
    ng = cv2.imread(os.path.join(root, bg_id, TYPE, 'defect', name))
    return ok, ng, gt

def find_corr_img_ori(bg_id, img_path):
    root = r'/home/hc/lby/ccd/mmsegmentation/data/SynLCD/Possion'
    TYPE = 'mixed_hard'
    name = img_path.split('/')[-1].split('_')[-1]
    ok = cv2.imread(os.path.join(root, bg_id.lstrip('bg') + '.png'))
    ng = cv2.imread(os.path.join(root, bg_id, TYPE, 'defect', name))
    gt = cv2.imread(os.path.join(root, bg_id, TYPE, 'gt', name))
    gt[gt != 0] = 255
    return ok, ng, gt

def find_type(path):
    if 'line' in path:
        TYPE = 'line'
    elif 'abpt' in path:
        TYPE = 'abpt'
    else:
        TYPE = 'mixed_hard'
    return TYPE

def find_our_best(name_list):
    bgs = list(name_list[0].keys())
    mnum = len(name_list)
    inum = len(name_list[0][bgs[0]]['ng'])
    err_list = {k:np.zeros((mnum,inum))
                for k in bgs}
    best_list = {}
    for midx, method in enumerate(name_list):
        for bid, bg in enumerate(bgs):
            for pid, img_path in enumerate(method[bg]['ng']):
                img = cv2.imread(img_path, -1).sum(axis=2)
                err_list[bg][midx][pid] = np.sum(img == 255)
            best_list[bg] = np.argsort(err_list[bg][0,:])
    return err_list, best_list


def extract_img(name_list, num=6):
    img_list = []
    keys = list(name_list[0].keys())
    print('Finding our best...')
    err_list, best_list = find_our_best(name_list)
    print('Extracting image...')
    Errs = []
    for i in range(num):
        bg_id = keys[i // (num // len(keys))]

        # TODO: automatically select the results where our method is at the first place
        #pop_id = random.randint(0, len(name_list[0][bg_id]['ng']) - 1)
        pop_id = best_list[bg_id][i]
        ok, ng, gt = find_corr_img(bg_id, name_list[0][bg_id]['ng'][pop_id])
        tmp = [ok, ng, gt]
        for method in name_list:
            img_path = method[bg_id]['ng'][pop_id]
            tmp.append(cv2.imread(img_path))
        img_list.append(tmp)
        Errs.append(err_list[bg_id][:,pop_id])
    return img_list,Errs

def show1(img_list, name=[], save_dir='test.png'):

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


def show(img_list, Errs=None, names=[], save_dir='compare_synlcd.svg'):
    print('Drawing picture...')
    rows = len(img_list)
    cols = len(img_list[0])
    f, ax = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    plt.tight_layout()
    plt.subplots_adjust(wspace=1,hspace=1)
    f.tight_layout()
    pos_list = [[210,300], [270,300], [100,480], [50,480], [50,460], [220,100]]
    for row_id in range(rows):
        for col_id in range(cols):
            image = img_list[row_id][col_id][:, :, ::-1]
            ax[row_id, col_id].imshow(image)
            ax[row_id, col_id].set_xticks([])
            ax[row_id, col_id].set_yticks([])
            if col_id>=3:
                ax[row_id, col_id].text(pos_list[row_id][0], pos_list[row_id][1],
                                        'Err:%d'%int(Errs[row_id][col_id-3]),
                                        color='gold', fontsize=14)
    for i in range(ax.shape[1]):
        ax[row_id, i].set_xlabel(names[i], fontsize=30)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    #plt.show()
    plt.savefig(save_dir, dpi=300, pad_inches=0.1, bbox_inches='tight', format='svg')
    plt.close()


if __name__ == '__main__':
    roots = [r'work_dirs/siamese_segformer_synlcd_cad/',#mixed_diffFPN_v11/
             r'work_dirs/segformer/eval_output',
             r'work_dirs/ocrnet/eval_output',
             r'work_dirs/danet/eval_output',
             r'work_dirs/deeplabv3plus/eval_output',
             r'work_dirs/pspnet/eval_output',
             r'work_dirs/FCN_synLCD/eval_output'
             ]

    name_list = [extract_list(os.path.join(root, 'mixed_output')) for root in roots]
    img_list,Errs = extract_img(name_list)
    show(img_list, Errs,
         names=['OK', 'NG', 'GT', 'Our', 'Segformer', 'OCRNet', 'DANet', 'Deeplab', 'PSPNet', 'FCN'])
    print('Complete!')
    for _ in Errs:
        print(_)