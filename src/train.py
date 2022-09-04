'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-05-31 15:54:02
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-08-17 15:39:04
FilePath: /low_to_high/pytorch-3dunet/pytorch3dunet/train.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import imp
import random

import torch

from unet3d.config import load_config
from unet3d.trainer import create_trainer
from unet3d.trainer_joint import create_joint_trainer
from unet3d.trainer_ARNet_GAN import create_ARNet_GAN_trainer
from unet3d.utils import get_logger

logger = get_logger('TrainingSetup')


def main():
    # Load and log experiment configuration
    config = load_config()
    logger.info(config)

    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        logger.warning('Using CuDNN deterministic setting. This may slow down the training!')
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True

    # create trainer
    trainer = create_ARNet_GAN_trainer(config)
    # trainer = create_joint_trainer(config)
    # trainer = create_trainer(config)
    # Start training
    trainer.fit()


if __name__ == '__main__':
    main()
