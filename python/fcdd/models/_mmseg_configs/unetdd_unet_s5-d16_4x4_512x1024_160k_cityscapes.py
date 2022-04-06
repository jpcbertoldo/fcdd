"""
Based on mmsegmentation/configs/unet/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes.py
"""

configs_base_path = "../../../mmsegmentation/configs"

_base_ = [
    # f'{configs_base_path}/_base_/models/unetdd_unet_s5-d16.py', 
    f'{configs_base_path}/_base_/models/fcn_unet_s5-d16.py', 
    f'{configs_base_path}/_base_/datasets/cityscapes.py',
    f'{configs_base_path}/_base_/default_runtime.py', 
    f'{configs_base_path}/_base_/schedules/schedule_160k.py'
]

norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    # i left the parts between --- as they are to avoid breaking things
    # auxiliary_head=dict(num_classes=19),  # replaced
    # ----------------------------------------------------------------------------
    # decode_head=dict(num_classes=19),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
    # ----------------------------------------------------------------------------
    auxiliary_head=None,
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(
        type='HSCHead',
        channels=64,
        norm_cfg=norm_cfg,
        # conv_dd_num_logits=1,
    )
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
