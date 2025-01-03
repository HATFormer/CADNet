_base_ = [
    '../_base_/models/danet_r50-d8.py',
    '../_base_/datasets/pascal_voc12_aug.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
model = dict(
    decode_head=dict(num_classes=21), auxiliary_head=dict(num_classes=21))
    
work_dir = './work_dirs/danet'
