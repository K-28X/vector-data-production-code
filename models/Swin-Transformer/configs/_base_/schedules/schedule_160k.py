# # optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
# optimizer_config = dict()
# # learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)


# # optimizer
optimizer = dict(type='Adam', lr=5e-5, weight_decay=0.0005)
optimizer_config = dict()

# learning policy
lr_config = dict(
    policy='poly',  # 或 'CosineAnnealing'，按你实际框架支持选择
    power=0.9,
    min_lr=1e-6,
    by_epoch=True
)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=16000, metric='mIoU', pre_eval=True)
