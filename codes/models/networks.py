import torch
import models.archs.discriminator_vgg_arch as SRGAN_arch
import models.archs.FastNet_arch as FastNet_arch

# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    # image restoration
    if which_model == 'FastNet':
        netG = FastNet_arch.FastNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG


# Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'],
                                                is_depthwise_separable_convolutions='is_depthwise_separable_convolutions' in opt_net and opt_net['is_depthwise_separable_convolutions'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


# Define network used for perceptual loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    if (not 'network_F' in opt) or opt['network_F']['which_model_F'] == 'VGG':
        # PyTorch pretrained VGG19-54, before ReLU.
        if use_bn:
            feature_layer = 49
        else:
            feature_layer = 34
        netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                              use_input_norm=True, device=device)
    elif opt['network_F']['which_model_F'] == 'squeezenet':
        netF = SRGAN_arch.SqueezeNetFeatureExtractor(use_input_norm=True, device=device)
    elif opt['network_F']['which_model_F'] == 'mobilenet':
        # raise NotImplementedError
        netF = SRGAN_arch.MobileNetFeatureExtractor(use_input_norm=True, device=device)
    elif opt['network_F']['which_model_F'] == 'shufflenet':
        # raise NotImplementedError
        netF = SRGAN_arch.ShuffleNetFeatureExtractor(use_input_norm=True, device=device)
    else:
        raise NotImplementedError

    netF.eval()  # No need to train
    return netF
