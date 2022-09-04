import importlib
import os

import torch
import torch.nn as nn

from datasets.utils import get_test_loaders
from unet3d import utils
from unet3d.config import load_config
from unet3d.model import get_model, define_G

logger = utils.get_logger('UNet3DPredict')


# def _get_predictor(model, output_dir, config):
#     predictor_config = config.get('predictor', {})
#     class_name = predictor_config.get('name', 'StandardPredictor')

#     # m = importlib.import_module('unet3d.predictor')
#     m = importlib.import_module('unet3d.predictor')
#     predictor_class = getattr(m, class_name)

#     return predictor_class(model, output_dir, config, **predictor_config)


# def main():
#     # Load configuration
#     config = load_config()

#     # Create the model
#     model = get_model(config['model'])

#     # Load model state
#     model_path = config['model_path']
#     logger.info(f'Loading model from {model_path}...')
#     utils.load_checkpoint(model_path, model)
#     # use DataParallel if more than 1 GPU available
#     device = config['device']
#     if torch.cuda.device_count() > 1 and not device.type == 'cpu':
#         model = nn.DataParallel(model)
#         logger.info(f'Using {torch.cuda.device_count()} GPUs for prediction')

#     logger.info(f"Sending the model to '{device}'")
#     model = model.to(device)

#     output_dir = config['loaders'].get('output_dir', None)
#     if output_dir is not None:
#         os.makedirs(output_dir, exist_ok=True)
#         logger.info(f'Saving predictions to: {output_dir}')

#     # create predictor instance
#     predictor = _get_predictor(model, output_dir, config)

#     for test_loader in get_test_loaders(config):
#         # run the model prediction on the test_loader and save the results in the output_dir
#         predictor(test_loader)

def _get_predictor(refine_model, model, output_dir, config):
    predictor_config = config.get('predictor', {})
    class_name = predictor_config.get('name', 'StandardPredictor')

    # m = importlib.import_module('unet3d.predictor')
    m = importlib.import_module('unet3d.predictor_joint')
    predictor_class = getattr(m, class_name)

    return predictor_class(refine_model, model, output_dir, config, **predictor_config)

def main():
    # Load configuration
    config = load_config()

    # Create the model
    # model = get_model(config['model'])
    model = define_G(**config['model'])
    refine_model = get_model(config['refine_model'])
    # refine_model_train = get_model(config['refine_model'])

    # Load model state
    model_path = config['model_path']
    logger.info(f'Loading model from {model_path}...')
    utils.load_checkpoint(model_path, model)

    refine_model_path = config['refine_model_path']
    logger.info(f'Loading refine model from {refine_model_path}...')
    utils.load_checkpoint(refine_model_path, refine_model)

    # refine_model_train_path = config['refine_model_train_path']
    # logger.info(f'Loading refine model from {refine_model_train_path}...')
    # utils.load_checkpoint(refine_model_train_path, refine_model_train)
    
    # Load device
    device = config['device']

    # Send models to device
    logger.info(f"Sending the models to '{device}'")
    model = model.to(device)
    refine_model = refine_model.to(device)
    # refine_model_train = refine_model_train.to(device)

    # Create output path
    output_dir = config['loaders'].get('output_dir', None)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f'Saving predictions to: {output_dir}')
    
    # Create predictor instance
    predictor = _get_predictor(refine_model, model, output_dir, config)

    for test_loader in get_test_loaders(config):
        # run the model prediction on the test_loader and save the results in the output_dir
        predictor(test_loader) 



if __name__ == '__main__':
    main()