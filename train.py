"""
Based on https://github.com/ZikangZhou/QCNet/blob/main/train_qcnet.py
"""


from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from datamodules import WaymoDataModule
from models import UniMM

try:
    import resource
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(655360, hard_limit), hard_limit))
except (ImportError, OSError, ValueError):
    pass


if __name__ == '__main__':
    pl.seed_everything(2025, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--val_batch_size', type=int, required=True)
    parser.add_argument('--test_batch_size', type=int, required=True)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, required=True)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--experiment', type=str, default='test')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--smoke_test', action='store_true',
                        help='Run one epoch with 20 train and 2 validation batches')
    UniMM.add_model_specific_args(parser)
    args = parser.parse_args()

    model = UniMM(**vars(args))
    datamodule = {
        'waymo': WaymoDataModule,
        'bench2drive': WaymoDataModule,
    }[args.dataset](**vars(args))

    model_checkpoint = ModelCheckpoint(monitor='val_cls_loss', save_top_k=5, save_last=True, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(
        accelerator=args.accelerator, devices=args.devices,
        strategy=DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)
                 if args.devices > 1 else 'auto',
        callbacks=[lr_monitor] if args.smoke_test else [model_checkpoint, lr_monitor],
        enable_checkpointing=not args.smoke_test,
        accumulate_grad_batches=args.accumulate_grad_batches,
        max_epochs=1 if args.smoke_test else args.max_epochs,
        limit_train_batches=20 if args.smoke_test else 1.0,
        limit_val_batches=2 if args.smoke_test else 1.0,
        num_sanity_val_steps=0 if args.smoke_test else 2,
        log_every_n_steps=1 if args.smoke_test else 50,
        detect_anomaly=args.smoke_test,
        default_root_dir=f'{args.output_dir}/{args.experiment}'
    )
    
    trainer.fit(model, datamodule, ckpt_path=args.ckpt_path)
