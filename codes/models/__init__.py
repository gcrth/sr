import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']
    # image restoration
    if model == 'sr':  # PSNR-oriented super resolution
        if 'attack' not in opt:
            from .SR_model import SRModel as M
        else:
            from .SR_model_adv import SRModel as M
    elif model == 'srgan':  # GAN-based super resolution, SRGAN / ESRGAN
        if 'attack' not in opt:
            from .SRGAN_model import SRGANModel as M
        else:
            from .SRGAN_model_adv import SRGANModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
