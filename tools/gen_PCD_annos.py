import os
import cv2
from tqdm import tqdm
import numpy as np
from skimage.draw import polygon2mask

anno_dir = '../../data/PCB_DATASET/Annotations'
out_dir = '../../data/PCB_DATASET/seg_annos'
ref_dir = '../../data/PCB_DATASET/PCB_USED'
image_dir = '../../data/PCB_DATASET/images'
sets = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Spur', 'Short', 'Spurious_copper']
#sets = ['Mouse_bite', 'Open_circuit']

def genFilter(anno_file,ref):
    boxes = parse_xml(anno_file)
    mask = np.zeros_like(ref)
    for box in boxes:
        xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']
        mask[ymin - 5:ymax + 5, xmin - 5:xmax + 5] = 1
    return mask

def genAnno():
    ref_names = os.listdir(ref_dir)
    ref_images = {x.strip('.jpg'):cv2.imread(os.path.join(ref_dir,x), cv2.IMREAD_GRAYSCALE)
                  for x in ref_names}
    for st_idx, st in enumerate(sets):
        path = os.path.join(image_dir,st)
        names = os.listdir(path)
        names = [x for x in names if '.jpg' in x]
        names.sort(key=lambda x: [int(x.split('_')[0]),int(x.split('_')[-1].strip('.jpg'))])
        out_path = os.path.join(out_dir,st)
        os.makedirs(out_path,exist_ok=True)
        for name in tqdm(names):
            ref_image = ref_images[name.split('_')[0]]
            defect_image = cv2.imread(os.path.join(image_dir, st, name), cv2.IMREAD_GRAYSCALE)

            if st not in ['Mouse_bite', 'Open_circuit']:
                anno = np.float16(ref_image)-np.float16(defect_image)

                anno[np.abs(anno)<5]=0
                #anno[anno!=0] = 255
                h, w = anno.shape
                #gray_anno = cv2.cvtColor(np.uint8(anno),cv2.COLOR_RGB2GRAY)
                _, binary = cv2.threshold(np.uint8(anno), h, w, cv2.THRESH_OTSU)
                binary[binary != 0] = st_idx+1

                binary *= genFilter(os.path.join(anno_dir, st, name.replace('jpg', 'xml')), binary)

                binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, np.ones((5,5),np.uint8))
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                binary[binary != 0] = st_idx + 1
            else:
                binary = np.zeros_like(defect_image)
                H, W = binary.shape
                polygon_list = rectangle2polygon(os.path.join(anno_dir, st, name.replace('jpg', 'xml')))
                for polygon in polygon_list:
                    anno = polygon2mask((H, W), polygon[:, (1, 0)])
                    binary[anno!=0] = anno[anno!=0]
                binary[binary != 0] = st_idx+1

            print(np.unique(binary))
            cv2.imwrite(os.path.join(out_path,name.replace('jpg','png')), binary)

import xml.etree.ElementTree as ET

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    box_list = []

    for obj in root.findall('.//object'):
        name = obj.find('name').text
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)

        box_info = {'name': name, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        box_list.append(box_info)

    return box_list

def rectangle2polygon(path):
    boxes = parse_xml(path)
    output = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']
        points = [(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)]
        output.append(np.array(points))
    return output

def genRef(out_dir='../../data/PCB_DATASET/PCB_USED'):
    st = sets[0]
    path = os.path.join(image_dir,st)
    ref_names = ['{:02d}'.format(num) for num in range(1,13)]
    names = os.listdir(path)
    names = [x for x in names if '.jpg' in x]
    names.sort(key=lambda x: [int(x.split('_')[0]), int(x.split('_')[-1].strip('.jpg'))])
    for idx, name in enumerate(names):
        pre = name.split('_')[0]
        if pre in ref_names:
            next_idx = idx+2
            if pre=='12':
                next_idx = idx+9
            print(name,names[next_idx])
            ref_names.remove(pre)
            boxes = parse_xml(os.path.join(anno_dir, st, name.replace('jpg','xml')))
            img = cv2.imread(os.path.join(path, name))
            next_img = cv2.imread(os.path.join(path, names[next_idx]))
            for box in boxes:
                xmin, ymin, xmax, ymax = box['xmin'],box['ymin'],box['xmax'],box['ymax']
                img[ymin - 5:ymax + 5, xmin - 5:xmax + 5, :] = next_img[ymin - 5:ymax + 5, xmin - 5:xmax + 5, :]
                #cv2.rectangle(img, (xmin-5, ymin+5), (xmax+5, ymax+5), (0, 255, 0), 2)

            cv2.imwrite(os.path.join(out_dir, pre + '.JPG'), img)
        else:
            continue



def showAnno(out_dir='../../data/PCB_DATASET/split_showanno',
             seg_anno_dir='../../data/PCB_DATASET/split_seg_annos',
             image_dir='../../data/PCB_DATASET/split_images'):
    for st in sets:
        path = os.path.join(image_dir,st)
        names = os.listdir(path)
        names = [x for x in names if '.jpg' in x]
        names.sort(key=lambda x: [int(x.split('_')[0]),int(x.split('_')[-1].strip('.jpg'))])
        out_path = os.path.join(out_dir,st)
        os.makedirs(out_path, exist_ok=True)
        for name in tqdm(names):
            img = cv2.imread(os.path.join(image_dir, st, name))
            anno = cv2.imread(os.path.join(seg_anno_dir, st, name.replace('.jpg','.png')))
            img[anno[:,:,-1]!=0] = (0,0,255)

            # boxes = parse_xml(os.path.join(anno_dir, st, name.replace('jpg','xml')))
            # for box in boxes:
            #     xmin, ymin, xmax, ymax = box['xmin'],box['ymin'],box['xmax'],box['ymax']
            #     cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(out_path,name),img)


# genRef()
#genAnno()
showAnno()