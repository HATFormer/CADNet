import cv2
import random
import matplotlib.pyplot as plt
import numpy as np
import os


def extract_list(root, dirs):
    img_dict = {'bg8': {_: [] for _ in dirs},
                'bg9': {_: [] for _ in dirs},
                'bg10': {_: [] for _ in dirs}}
    names = os.listdir(os.path.join(root,dirs[0]))
    for name in names:
        pre, idx = name.split('_')
        for dir in dirs:
            if int(idx.strip('.png')) <= 300:
                img_dict[pre][dir].append(os.path.join(root, dir, name))
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

def find_our_best(name_list):
    bgs = list(name_list[0].keys())
    mnum = len(name_list)
    inum = len(name_list[0][bgs[0]]['pred'])
    err_list = {k:np.zeros((mnum,inum))
                for k in bgs}
    best_list = {}
    for midx, method in enumerate(name_list):
        for bid, bg in enumerate(bgs):
            for pid, img_path in enumerate(method[bg]['pred']):
                try:
                    img = cv2.imread(img_path, -1).sum(axis=2)
                except:
                    img = cv2.imread(img_path.replace('pred','mixed_output'), -1).sum(axis=2)
                err_list[bg][midx][pid] = np.sum(img == 255)
    for bid, bg in enumerate(bgs):
        err_sort = np.argsort(err_list[bg],axis=0)
        target = np.ones_like(err_sort)
        target[0, :] *= 2
        target[1, :] *= 1
        target[2, :] *= 0
        result = np.sum(err_sort == target,axis=0)
        best_list[bg]=np.argwhere(result==3)
    return err_list, best_list

def extract_img(name_list, num=[0,1,2]):
    img_list = []
    keys = list(name_list[0].keys())
    print('Finding our best...')
    err_list, best_list = find_our_best(name_list)
    print('Extracting image...')
    Errs = []
    for i in num:

        bg_id = keys[i]

        pop_id = int(best_list[bg_id][-1])
        ok, ng, gt = find_corr_img(bg_id, name_list[-1][bg_id]['pred'][pop_id])
        tmp = [ng,gt]
        for midx in range(len(name_list)):
            #if midx!=2:
            try:
                img = cv2.imread(name_list[midx][bg_id]['pred'][pop_id])
                _ = img.shape
            except:
                img = cv2.imread(name_list[midx][bg_id]['pred'][pop_id].replace('pred','mixed_output'))
            tmp.append(img)

        tmp.append(cv2.imread(name_list[-1][bg_id]['fea'][pop_id]))
        tmp.append(cv2.imread(name_list[-1][bg_id]['prob'][pop_id]))
        for midx in range(len(name_list)):
            #if midx != 2:
            tmp.append(cv2.imread(name_list[midx][bg_id]['distmap'][pop_id]))


        img_list.append(tmp)
        Errs.append(err_list[bg_id][:,pop_id])
    return img_list, Errs


def show1(img_list, name=[], save_dir='ablation.svg'):
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

    plt.savefig(save_dir, dpi=300, pad_inches=0.1, bbox_inches='tight', format='svg')
    plt.close()


def show(img_list, names=[], save_dir='test.png'):
    rows = len(img_list)*2
    cols = len(img_list[0])//2
    f, ax = plt.subplots(rows, cols, figsize=(cols, rows+0.4))
    plt.tight_layout()
    plt.subplots_adjust(wspace=1,hspace=1)
    f.tight_layout()
    pos_list = [[210,495],[30,460],[30,480],[30,480],[30,460],[30,480]]
    for i,sub_list in enumerate(img_list):
        for row_id in range(2):
            for col_id in range(cols):
                image = sub_list[row_id*cols+col_id][:, :, ::-1]
                ax[row_id+2*i, col_id].imshow(image)
                ax[row_id+2*i, col_id].set_xticks([])
                ax[row_id+2*i, col_id].set_yticks([])

                ax[row_id+2*i, col_id].set_xlabel(names[row_id*cols+col_id], fontsize=5)
                if row_id==0 and col_id >= 2:
                    ax[row_id+2*i, col_id].text(310,350,
                                            'Err:%d' % int(Errs[i][col_id - 2]),
                                            color='gold', fontsize=4)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(save_dir, dpi=300, pad_inches=0.05, bbox_inches='tight')
    #plt.show()
    plt.close()


if __name__ == '__main__':
    roots = [r'work_dirs/siameseCGNet_pretrained111/mixed_diffFPN_v2/',
             r'work_dirs/siameseCGNet_pretrained111/mixed_diffFPN_v3/weight200/double3090/',
             #r'work_dirs/siameseCGNet_pretrained111/mixed_diffFPN_v4/',
             r'work_dirs/siameseCGNet_pretrained111/mixed_diffFPN_v11/']
    dirs = ['distmap','fea','prob','pred']
    name_list = [extract_list(os.path.join(root, 'output'),dirs) for root in roots]
    img_list,Errs = extract_img(name_list,num=[0,2])
    show(img_list, names=['input', 'GT', 'pred_noCL','pred_CL','pred_BCL','prob_noCAD','prob_CAD',
                          'distmap_noCL','distmap_CL','distmap_BCL'])
    print(1)
