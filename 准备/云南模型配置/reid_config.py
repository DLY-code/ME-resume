backend_args = None
data_root = '/data/DLY/YunNan/mmdetection/datasets/VeRi-YunNan-resample'
dataset_type = 'ReIDDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=5, max_keep_ckpts=10, save_best='auto',
        type='CheckpointHook'),
    logger=dict(interval=1, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = '/data/DLY/YunNan/mmdetection/work_dirs/reid_resnet101_256x256_20260116_e60/best_reid-metric_mAP_epoch_37.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        depth=101,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        type='mmpretrain.ResNet'),
    data_preprocessor=dict(
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        to_rgb=True,
        type='ReIDDataPreprocessor'),
    head=dict(
        act_cfg=dict(type='ReLU'),
        fc_channels=1024,
        in_channels=2048,
        loss_cls=dict(loss_weight=1.0, type='mmpretrain.CrossEntropyLoss'),
        loss_triplet=dict(loss_weight=0.5, margin=0.3, type='TripletLoss'),
        norm_cfg=dict(type='BN1d'),
        num_classes=769,
        num_fcs=1,
        out_channels=128,
        type='LinearReIDHead'),
    init_cfg=dict(
        checkpoint=
        '/data/DLY/YunNan/mmdetection/resnet101_8xb32_in1k_20210831-539c63f8.pth',
        type='Pretrained'),
    neck=dict(kernel_size=(
        8,
        8,
    ), stride=1, type='GlobalAveragePooling'),
    type='BaseReID')
optim_wrapper = dict(
    clip_grad=dict(max_norm=10.0, norm_type=2),
    loss_scale='dynamic',
    optimizer=dict(lr=0.00035, type='Adam', weight_decay=0.0005),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=True, end=5, start_factor=0.01, type='LinearLR'),
    dict(
        T_max=55,
        begin=5,
        by_epoch=True,
        end=60,
        eta_min=1e-07,
        type='CosineAnnealingLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='meta/test.txt',
        data_prefix=dict(img_path='imgs'),
        data_root='/data/DLY/YunNan/mmdetection/datasets/VeRi-YunNan-resample',
        pipeline=[
            dict(backend_args=None, to_float32=True, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                256,
                256,
            ), type='Resize'),
            dict(
                pad_val=dict(img=(
                    128,
                    128,
                    128,
                )),
                size=(
                    256,
                    256,
                ),
                type='Pad'),
            dict(type='PackReIDInputs'),
        ],
        triplet_sampler=None,
        type='ReIDDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    metric=[
        'mAP',
        'CMC',
    ], type='ReIDMetrics')
test_pipeline = [
    dict(backend_args=None, to_float32=True, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        256,
        256,
    ), type='Resize'),
    dict(pad_val=dict(img=(
        128,
        128,
        128,
    )), size=(
        256,
        256,
    ), type='Pad'),
    dict(type='PackReIDInputs'),
]
train_cfg = dict(max_epochs=60, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='meta/train.txt',
        data_prefix=dict(img_path='imgs'),
        data_root='/data/DLY/YunNan/mmdetection/datasets/VeRi-YunNan-resample',
        pipeline=[
            dict(
                share_random_params=False,
                transforms=[
                    dict(
                        backend_args=None,
                        to_float32=True,
                        type='LoadImageFromFile'),
                    dict(keep_ratio=True, scale=(
                        256,
                        256,
                    ), type='Resize'),
                    dict(
                        pad_val=dict(img=(
                            128,
                            128,
                            128,
                        )),
                        size=(
                            256,
                            256,
                        ),
                        type='Pad'),
                    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
                    dict(
                        brightness_delta=32,
                        contrast_range=(
                            0.5,
                            1.5,
                        ),
                        hue_delta=18,
                        saturation_range=(
                            0.5,
                            1.5,
                        ),
                        type='PhotoMetricDistortion'),
                ],
                type='TransformBroadcaster'),
            dict(
                meta_keys=(
                    'flip',
                    'flip_direction',
                ), type='PackReIDInputs'),
        ],
        triplet_sampler=dict(ins_per_id=4, num_ids=4),
        type='ReIDDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        share_random_params=False,
        transforms=[
            dict(backend_args=None, to_float32=True, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                256,
                256,
            ), type='Resize'),
            dict(
                pad_val=dict(img=(
                    128,
                    128,
                    128,
                )),
                size=(
                    256,
                    256,
                ),
                type='Pad'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(
                brightness_delta=32,
                contrast_range=(
                    0.5,
                    1.5,
                ),
                hue_delta=18,
                saturation_range=(
                    0.5,
                    1.5,
                ),
                type='PhotoMetricDistortion'),
        ],
        type='TransformBroadcaster'),
    dict(meta_keys=(
        'flip',
        'flip_direction',
    ), type='PackReIDInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='meta/test.txt',
        data_prefix=dict(img_path='imgs'),
        data_root='/data/DLY/YunNan/mmdetection/datasets/VeRi-YunNan-resample',
        pipeline=[
            dict(backend_args=None, to_float32=True, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                256,
                256,
            ), type='Resize'),
            dict(
                pad_val=dict(img=(
                    128,
                    128,
                    128,
                )),
                size=(
                    256,
                    256,
                ),
                type='Pad'),
            dict(type='PackReIDInputs'),
        ],
        triplet_sampler=None,
        type='ReIDDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    metric=[
        'mAP',
        'CMC',
    ], type='ReIDMetrics')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/reid_resnet101_256x256_20260116_e60'
