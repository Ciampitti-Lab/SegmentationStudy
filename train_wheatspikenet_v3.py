import os
import argparse
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/wheatspikenet_v3.py', help='path to config')
    parser.add_argument('--work-dir', default=None, help='override work_dir')
    parser.add_argument('--resume', default=None, help='checkpoint to resume from')
    parser.add_argument('--amp', action='store_true', help='enable mixed precision')
    parser.add_argument("--cfg-options", nargs='+', action=DictAction, help='override settings', default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    os.makedirs(cfg.work_dir, exist_ok=True)
    if args.amp:
        cfg.setdefault('optim_wrapper', {})
        cfg.optim_wrapper.setdefault('type', 'AmpOptimWrapper')
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')
    if args.resume is not None:
        cfg.load_from = args.resume
        cfg.resume = True
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main()
