import os
import shutil

SRC = r'/home/hc/lby/data/Possion'
DST = r'/media/hc/3A5664FB5664B8F1/lby/defect/SynLCD'
TYPE = 'line'

for i in range(1,11):
    print('Copying file from bg%d..'%i)
    root1 = os.path.join(SRC,'bg{}/{}/defect'.format(i,TYPE))
    root2 = os.path.join(SRC, 'bg{}/{}/gt'.format(i, TYPE))
    dst1 = os.path.join(DST, 'bg{}/{}/defect'.format(i, TYPE))
    dst2 = os.path.join(DST, 'bg{}/{}/gt'.format(i, TYPE))
    os.makedirs(dst1,exist_ok=True)
    os.makedirs(dst2,exist_ok=True)
    names = os.listdir(root1)
    for name in names:
        shutil.copyfile(os.path.join(root1,name),os.path.join(dst1,name))
        shutil.copyfile(os.path.join(root2, name), os.path.join(dst2, name))
print('Done!')