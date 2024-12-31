# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
import warnings
import torch.nn.functional as F
import catboost.eval
import mmcv
import numpy as np
import torch
from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
import cv2
import matplotlib.pyplot as plt
from tools.utils import feamap_handler, visualize_tsne
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM


def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-10)

def tensor2img(img, color=False):

    img = img.cpu().detach().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-10)
    img = np.uint8(img * 255)
    if color:
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img


def gradCAM_vis(model, target_layers, input_tensor, gt,
                rgb_img=None, out_path=None, img_path=None, categories=[1, 2]):
    class SemanticSegmentationTarget:
        def __init__(self, category, mask):
            self.category = category
            self.mask = torch.from_numpy(mask)
            if torch.cuda.is_available():
                self.mask = self.mask.cuda()

        def __call__(self, model_output):
            if model_output.shape[1]>1:
                return (model_output[0, self.category, :, :]).sum()
            else:
                return (model_output[0, 0, :, :] * self.mask).sum()

    gt_float = np.float32(gt == 255)

    img = np.zeros_like(gt_float)
    for category in categories:
        targets = [SemanticSegmentationTarget(category, gt_float)]
        with GradCAM(model=model,
                     target_layers=target_layers,
                     use_cuda=torch.cuda.is_available()) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            # cam_image = show_cam_on_image(normalize(rgb_img), grayscale_cam[0], use_rgb=True)
            # img_c1 = cv2.applyColorMap(np.uint8(grayscale_cam[1] * 255), cv2.COLORMAP_JET)
            img += normalize(grayscale_cam[0])

    img = np.clip(img, a_min=0., a_max=1.)
    img = cv2.applyColorMap(np.uint8(img * 255), cv2.COLORMAP_JET)

    if out_path is not None and img_path is not None:
        cv2.imwrite(os.path.join(out_path, get_name(img_path) + '_c0.png'), img)
        # cv2.imwrite(os.path.join(out_path,get_name(img_path)+'_c1.png'),img_c1)

    else:
        return img


def get_name(img_path):
    idx = img_path.split('/')[-1].strip('.png')
    bg = [_ for _ in img_path.split('/') if 'bg' in _][-1][2:]
    if 'non' in img_path:
        name = 'bg{}_{}'.format(bg, int(idx) + 300)
    else:
        name = 'bg{}_{}'.format(bg, idx)
    return name

def get_name_pcb(img_path):

    return img_path.split('/')[-1].strip('.jpg')

def get_name_mvtec(img_path):
    splits = img_path.split('/')
    return splits[-4:]

def my_show_prob(pred, img_path, out_dir):
    pred = F.softmax(pred, dim=1)
    name = get_name(img_path)

    os.makedirs(out_dir, exist_ok=True)

    line_out = os.path.join(out_dir, 'line')
    abpt_out = os.path.join(out_dir, 'abpt')
    os.makedirs(line_out, exist_ok=True)
    os.makedirs(abpt_out, exist_ok=True)

    if int(name.split('_')[-1]) <= 300:
        line = line_out + '/' + name + '.png'
        abpt = abpt_out + '/' + name + '.png'
        line_img = tensor2img(pred[0][1], color=False)
        abpt_img = tensor2img(pred[0][2], color=False)
        cv2.imwrite(line, line_img)
        cv2.imwrite(abpt, abpt_img)

def my_show_mvtec_divide(ori_img, img_path, ref_img_path, pred, distmap, out_dir):
    names = get_name_mvtec(img_path)
    input = cv2.imread(img_path)
    ref = cv2.imread(ref_img_path)
    if 'good' in img_path:
        gt = np.zeros_like(pred)

    else:
        gt = cv2.imread(img_path.replace('/test', '/ground_truth').replace('.png', '_mask.png'), -1)

    try:
        gt = np.stack([gt, gt, gt], 2)
    except:
        import pdb;
        pdb.set_trace()
    gt = np.uint8(gt)
    gt[gt != 0] = 255

    pred = np.stack([pred, pred, pred], 2)
    pred = np.uint8(pred)
    pred[pred != 0] = 255


    if False:
        gt_pos, pred_pos = np.zeros_like(gt), np.zeros_like(gt)
        gt_pos[gt != 0] = 1
        pred_pos[pred != 0] = 1
        fn_pos = gt_pos - pred_pos == 1
        fp_pos = pred_pos - gt_pos == 1
        pred[:, :, 1][fn_pos[:, :, 0]] = 255
        pred[:, :, 0][fp_pos[:, :, 0]] = 0
        pred[:, :, 1][fp_pos[:, :, 0]] = 0
        pred[:, :, 2][fp_pos[:, :, 0]] = 255

    out = os.path.join(out_dir, names[0], names[1], names[2])
    os.makedirs(out, exist_ok=True)

    cv2.imwrite(out + '/' + names[3].replace('.png','_pred.png'), pred)
    cv2.imwrite(out + '/' + names[3].replace('.png','_distmap.png'), distmap)
    cv2.imwrite(out + '/' + names[3].replace('.png','_gt.png'), gt)
    cv2.imwrite(out + '/' + names[3].replace('.png','_input.png'), input)
    cv2.imwrite(out + '/' + names[3].replace('.png','_ref.png'), ref)
    # cv2.imwrite(out + '/' + names[3].replace('.png','_fea.png'), maps[1])
    # cv2.imwrite(out + '/' + names[3].replace('.png','_prob.png'), maps[2])




def my_show_mvtec(ori_img, img_path, pred, out_dir,
            maps=None, defect_color=False,
            compare_color=False, mode='eval', names=None):
    names = get_name_mvtec(img_path)

    if 'good' in img_path:
        gt = np.zeros_like(pred)
    else:
        gt = cv2.imread(img_path.replace('/test', '/ground_truth').replace('.png', '_mask.png'), -1)

    try:
        gt = np.stack([gt, gt, gt], 2)
    except:
        import pdb;pdb.set_trace()
    gt = np.uint8(gt)
    gt[gt != 0] = 255

    pred = np.stack([pred, pred, pred], 2)
    pred = np.uint8(pred)
    pred[pred != 0] = 255

    # compare and generate indicating colors for pred
    if False:
        gt_pos, pred_pos = np.zeros_like(gt), np.zeros_like(gt)
        gt_pos[gt != 0] = 1
        pred_pos[pred != 0] = 1
        fn_pos = gt_pos - pred_pos == 1
        fp_pos = pred_pos - gt_pos == 1
        pred[:, :, 1][fn_pos[:, :, 0]] = 255
        pred[:, :, 0][fp_pos[:, :, 0]] = 0
        pred[:, :, 1][fp_pos[:, :, 0]] = 0
        pred[:, :, 2][fp_pos[:, :, 0]] = 255

    os.makedirs(out_dir, exist_ok=True)
    if mode == 'eval':
        line_out = os.path.join(out_dir, 'line')
        abpt_out = os.path.join(out_dir, 'abpt')
        os.makedirs(line_out, exist_ok=True)
        os.makedirs(abpt_out, exist_ok=True)

        if int(name.split('_')[-1]) <= 300:
            line = line_out + '/' + name + '.png'
            abpt = abpt_out + '/' + name + '.png'
            line_img = np.zeros_like(pred)
            line_img[pred == 1] = 255
            abpt_img = np.zeros_like(pred)
            abpt_img[pred == 2] = 255
            cv2.imwrite(line, line_img)
            cv2.imwrite(abpt, abpt_img)
    elif mode == 'concat':
        out = os.path.join(out_dir, 'out')
        os.makedirs(out, exist_ok=True)

        plot_sample(ori_img, gt, pred, out + '/' + name + '.png', maps, names)
    elif mode == 'divide':
        pred_out = os.path.join(out_dir, 'pred')
        distmap_out = os.path.join(out_dir, 'distmap')
        fea_out = os.path.join(out_dir, 'fea')
        prob_out = os.path.join(out_dir, 'prob')
        #d_ca_out = os.path.join(out_dir, 'd_ca')
        for out in [pred_out, distmap_out, fea_out, prob_out]:
            os.makedirs(out, exist_ok=True)
        cv2.imwrite(pred_out + '/' + name + '.png', pred)
        cv2.imwrite(distmap_out + '/' + name + '.png', maps[0])
        cv2.imwrite(fea_out + '/' + name + '.png', maps[1])
        cv2.imwrite(prob_out + '/' + name + '.png', maps[2])
        #cv2.imwrite(d_ca_out + '/' + name + '.png', maps[3])
    elif mode == 'mixed_output':
        out = os.path.join(out_dir, names[0],names[1],names[2])
        os.makedirs(out, exist_ok=True)
        cv2.imwrite(out + '/' + names[3], pred)

def my_show(ori_img, img_path, pred, out_dir,
            maps=None, defect_color=False,
            compare_color=False, mode='eval', names=None):
    name = get_name(img_path)

    if 'non' in img_path:
        gt = np.zeros_like(pred)
    else:
        gt = cv2.imread(img_path.replace('/defect', '/gt'), -1)

    try:
        gt = np.stack([gt, gt, gt], 2)
    except:
        import pdb;pdb.set_trace()
    gt = np.uint8(gt)
    pred = np.stack([pred, pred, pred], 2)
    pred = np.uint8(pred)

    if defect_color:
        color_list = [[71, 130, 255], [255, 255, 0]]
    else:
        color_list = [[255, 255, 255], [255, 255, 255]]

    # generate color for pred and gt
    for cls_id, color in enumerate(color_list):
        for cid, cvalue in enumerate(color):
            pred[:, :, cid][pred[:, :, cid] == cls_id + 1] = cvalue
            gt[:, :, cid][gt[:, :, cid] == cls_id + 1] = cvalue

    # compare and generate indicating colors for pred
    if compare_color:
        gt_pos, pred_pos = np.zeros_like(gt), np.zeros_like(gt)
        gt_pos[gt != 0] = 1
        pred_pos[pred != 0] = 1
        fn_pos = gt_pos - pred_pos == 1
        fp_pos = pred_pos - gt_pos == 1
        pred[:, :, 1][fn_pos[:, :, 0]] = 255
        pred[:, :, 0][fp_pos[:, :, 0]] = 0
        pred[:, :, 1][fp_pos[:, :, 0]] = 0
        pred[:, :, 2][fp_pos[:, :, 0]] = 255

    os.makedirs(out_dir, exist_ok=True)
    if mode == 'eval':
        line_out = os.path.join(out_dir, 'line')
        abpt_out = os.path.join(out_dir, 'abpt')
        os.makedirs(line_out, exist_ok=True)
        os.makedirs(abpt_out, exist_ok=True)

        if int(name.split('_')[-1]) <= 300:
            line = line_out + '/' + name + '.png'
            abpt = abpt_out + '/' + name + '.png'
            line_img = np.zeros_like(pred)
            line_img[pred == 1] = 255
            abpt_img = np.zeros_like(pred)
            abpt_img[pred == 2] = 255
            cv2.imwrite(line, line_img)
            cv2.imwrite(abpt, abpt_img)
    elif mode == 'concat':
        out = os.path.join(out_dir, 'out')
        os.makedirs(out, exist_ok=True)

        plot_sample(ori_img, gt, pred, out + '/' + name + '.png', maps, names)
    elif mode == 'divide':
        pred_out = os.path.join(out_dir, 'pred')
        distmap_out = os.path.join(out_dir, 'distmap')
        fea_out = os.path.join(out_dir, 'fea')
        prob_out = os.path.join(out_dir, 'prob')
        #d_ca_out = os.path.join(out_dir, 'd_ca')
        for out in [pred_out, distmap_out, fea_out, prob_out]:
            os.makedirs(out, exist_ok=True)
        cv2.imwrite(pred_out + '/' + name + '.png', pred)
        cv2.imwrite(distmap_out + '/' + name + '.png', maps[0])
        cv2.imwrite(fea_out + '/' + name + '.png', maps[1])
        cv2.imwrite(prob_out + '/' + name + '.png', maps[2])
        #cv2.imwrite(d_ca_out + '/' + name + '.png', maps[3])
    elif mode == 'mixed_output':
        out = os.path.join(out_dir, 'mixed_output')
        os.makedirs(out, exist_ok=True)
        cv2.imwrite(out + '/' + name + '.png', pred)

def my_show_general(ori_img, img_path, pred, out_dir, refimg=None, compare_color=False):
    name = img_path.split('/')[-1].strip('.jpg')

    gt = cv2.imread(img_path.replace('/split_images', '/split_seg_annos').replace('jpg','png'), -1)

    gt = np.stack([gt, gt, gt], 2)
    gt = np.uint8(gt)
    pred = np.stack([pred, pred, pred], 2)
    pred = np.uint8(pred)

    color_list = [[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255],
                  [255, 255, 255], [255, 255, 255], [255, 255, 255]]

    # generate color for pred and gt
    for cls_id, color in enumerate(color_list):
        for cid, cvalue in enumerate(color):
            pred[:, :, cid][pred[:, :, cid] == cls_id + 1] = cvalue
            gt[:, :, cid][gt[:, :, cid] == cls_id + 1] = cvalue

    # compare and generate indicating colors for pred
    if compare_color:
        gt_pos, pred_pos = np.zeros_like(gt), np.zeros_like(gt)
        gt_pos[gt != 0] = 1
        pred_pos[pred != 0] = 1
        fn_pos = gt_pos - pred_pos == 1
        fp_pos = pred_pos - gt_pos == 1
        pred[:, :, 1][fn_pos[:, :, 0]] = 255
        pred[:, :, 0][fp_pos[:, :, 0]] = 0
        pred[:, :, 1][fp_pos[:, :, 0]] = 0
        pred[:, :, 2][fp_pos[:, :, 0]] = 255

    os.makedirs(out_dir+'/pred', exist_ok=True)
    os.makedirs(out_dir + '/img', exist_ok=True)
    os.makedirs(out_dir + '/gt', exist_ok=True)
    if refimg is not None:
        os.makedirs(out_dir + '/refimg', exist_ok=True)

    cv2.imwrite(out_dir + '/pred/' + name + '.png', pred)
    cv2.imwrite(out_dir + '/img/' + name + '.png', ori_img)
    cv2.imwrite(out_dir + '/refimg/' + name + '.png', refimg)
    cv2.imwrite(out_dir + '/gt/' + name + '.png', gt)

def plot_sample(image, label, segmentation, save_dir, maps=None, names=None):
    tot = 3 + len(maps) if maps is not None else 3
    plt.figure()
    plt.clf()
    plt.subplot(1, tot, 1)
    plt.xticks([])
    plt.yticks([])
    plt.title('Input')
    image = image[:, :, ::-1]
    if len(image.shape) > 2 and image.shape[2] == 1:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)

    plt.subplot(1, tot, 2)
    plt.xticks([])
    plt.yticks([])
    plt.title('GT')
    plt.imshow(label, cmap="gray")

    plt.subplot(1, tot, 3)
    plt.xticks([])
    plt.yticks([])
    plt.title('Pred')
    # display max
    vmax_value = max(1, np.max(segmentation))
    plt.imshow(segmentation, cmap="jet", vmax=vmax_value)

    if maps is not None:
        names = names if len(maps) == len(names) else ['map%d' % i for i in range(len(maps))]
        for idx, map in enumerate(maps):
            plt.subplot(1, tot, 3 + idx + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(names[idx])
            plt.imshow(cv2.cvtColor(map, cv2.COLOR_BGR2RGB), cmap="jet")

    plt.savefig(save_dir, bbox_inches='tight', dpi=300)
    plt.close()


def np2tmp(array, temp_file_name=None, tmpdir=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.
        tmpdir (str): Temporary directory to save Ndarray files. Default: None.
    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False, dir=tmpdir).name
    np.save(temp_file_name, array)
    return temp_file_name


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    opacity=0.5,
                    pre_eval=False,
                    format_only=False,
                    format_args={},
                    crosstest='none',
                    ori_model=None):
    """Test with single GPU by progressive mode.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.
    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    results = []
    cls_results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx
    loader_indices = data_loader.batch_sampler

    count = 0

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            # if count == 0:
            #     feamap_h = feamap_handler(model, data)
            name = data['img_metas'][0]._data[0][0]['filename']
            try:
                data['img_metas'][0]._data[0][0]['filename'] = '../..' + name.split('../..')[1]
                data['img_metas'][0]._data[0][0]['ref_filename'] = '../..' + name.split('../..')[2]
            except:
                pass
            raw_result = model(return_loss=False, **data)

            cls_result = None
            if raw_result[-1] == 'cls':
                raw_result, prob_result, cls_result, _ = raw_result
            elif raw_result[-1] == 'celoss' or 'contra' in raw_result[-1]:
                raw_result, prob_result, distance, _ = raw_result
            elif raw_result[-1] == 'other':
                raw_result, prob_result, _ = raw_result

        if efficient_test:
            raw_result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

        if format_only:
            raw_result = dataset.format_results(
                raw_result, indices=batch_indices, **format_args)

        if crosstest == 'line':
            for i, x in enumerate(raw_result):
                x[x != 0] = 1
                raw_result[i] = x
        elif crosstest == 'abpt':
            for i, x in enumerate(raw_result):
                x[x != 0] = 2
                raw_result[i] = x

        # for i, x in enumerate(raw_result):
        #     x[x != 0] = 4
        #     raw_result[i] = x
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            result = dataset.pre_eval(raw_result, indices=batch_indices)

        try:
            results.extend(result)
        except:
            results.extend(raw_result)
        if cls_result is not None:
            cls_results.extend(list(cls_result))

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])

            assert len(imgs) == len(img_metas)

            # show_feamaps, raw_feamaps = feamap_h.show_featuremap(data)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None
                MY_SHOW = 1 if out_dir else 0
                VIS_MODE = 'other'
                GAP = 1
                if count % GAP != 0:
                    pass
                elif MY_SHOW:
                    img_path = data['img_metas'][0]._data[0][0]['filename']
                    if 'non' in img_path:
                        gt = np.zeros((512, 512), dtype=np.uint8)
                    else:
                        gt = cv2.imread(img_path.replace('images', 'seg_annos').replace('jpg','png'), -1)

                        gt = cv2.resize(gt, dsize=(512, 512))#[:, :, 0]

                    gt[gt != 0] = 255
                    img_path = data['img_metas'][0]._data[0][0]['filename']

                    if VIS_MODE == 'feavis':
                        target_layers = [ori_model.backbone.conv0_0, ori_model.backbone.conv1_0,
                                         ori_model.backbone.conv2_0,
                                         ori_model.decode_head.att_head.conv_final,
                                         ori_model.decode_head.convs]
                        os.makedirs(os.path.join(out_dir, 'feavis'), exist_ok=True)
                        cv2.imwrite(os.path.join(out_dir, 'feavis', get_name(img_path) + '.png'), img_show)
                        for idx, target in enumerate(target_layers):
                            save_dir = os.path.join(out_dir, 'feavis', 'feamap%d' % idx)
                            os.makedirs(save_dir, exist_ok=True)
                            gradCAM_vis(ori_model, [target], img_show,
                                        torch.cat([img_tensor, refimg_tensor], dim=0),
                                        gt, save_dir, img_path)
                    elif VIS_MODE == 'prob':
                        my_show_prob(prob_result, data['img_metas'][0]._data[0][0]['filename'], out_dir)
                    elif VIS_MODE=='divide':
                        refimg_tensor = data['refimg'][0]
                        target_layers = [ori_model.decode_head.fusion_conv,
                                         ori_model.decode_head.convs[-2]]
                                         #ori_model.decode_head.att_head.convs]
                        tmp = []
                        for target in target_layers:
                            tmp.append(gradCAM_vis(ori_model, [target],
                                                   torch.cat([img_tensor, refimg_tensor], dim=0), gt))
                        distance = tensor2img(distance[0][0],color=True)
                        distance = cv2.resize(distance, dsize=(512, 512))
                        tmp.insert(0, distance)
                        my_show(img_show, data['img_metas'][0]._data[0][0]['filename'],
                                raw_result[-1], out_dir, maps=tmp, mode='divide', compare_color=False,
                                names=['features', 'scoremap'])
                    else:
                        # refimg_tensor = data['refimg'][0][0].permute(1,2,0)
                        # refimg = tensor2img(refimg_tensor, color=False)
                        # my_show_general(img_show, data['img_metas'][0]._data[0][0]['filename'],
                        #         raw_result[-1], out_dir, refimg=refimg, compare_color=True)
                        # distance = tensor2img(distance[0][0], color=True)
                        # distance = cv2.resize(distance, dsize=(512, 512))
                        my_show_mvtec(img_show, data['img_metas'][0]._data[0][0]['filename'],
                                      raw_result[-1], out_dir,
                                      maps=None, defect_color=False,
                                      compare_color=False, mode='mixed_output', names=None)
                        # my_show_mvtec_divide(img_show, data['img_metas'][0]._data[0][0]['filename'],
                        #                      data['img_metas'][0]._data[0][0]['ref_filename'],
                        #                     raw_result[-1], distance, out_dir)
                else:
                    model.module.show_result(
                        img_show,
                        result,
                        palette=dataset.PALETTE,
                        show=show,
                        out_file=out_file,
                        opacity=opacity)

        try:
            batch_size = len(result)
        except:
            batch_size = len(raw_result)
        for _ in range(batch_size):
            prog_bar.update()
        count += 1

    return results


def multi_gpu_test(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   efficient_test=False,
                   pre_eval=False,
                   format_only=False,
                   format_args={}):
    """Test model with multiple gpus by progressive mode.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. The same path is used for efficient
            test. Default: None.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.

    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    results = []
    dataset = data_loader.dataset
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx

    # batch_sampler based on DistributedSampler, the indices only point to data
    # samples of related machine.
    loader_indices = data_loader.batch_sampler

    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if efficient_test:
            result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

        if format_only:
            result = dataset.format_results(
                result, indices=batch_indices, **format_args)
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            result = dataset.pre_eval(result, indices=batch_indices)

        results.extend(result)

        if rank == 0:
            batch_size = len(result) * world_size
            for _ in range(batch_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results
