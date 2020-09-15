import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import GANLoss

logger = logging.getLogger('base')


class SRGANModel(BaseModel):
    def __init__(self, opt):
        super(SRGANModel, self).__init__(opt)

        train_opt = opt['train']

        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        self.netG = DataParallel(self.netG)

        self.netD = networks.define_D(opt).to(self.device)
        self.netD = DataParallel(self.netD)
        if self.is_train:
            self.netG.train()
            self.netD.train()

        if not self.is_train and 'attack' in self.opt:
            # G pixel loss
            if opt['pixel_weight'] > 0:
                l_pix_type = opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError(
                        'Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = opt['pixel_weight']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None

            # G feature loss
            if opt['feature_weight'] > 0:
                l_fea_type = opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError(
                        'Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = opt['feature_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                self.netF = networks.define_F(
                    opt, use_bn=False).to(self.device)
                self.netF = DataParallel(self.netF)

            # GD gan loss
            self.cri_gan = GANLoss(opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan_w = opt['gan_weight']

        self.delta = 0

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError(
                        'Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None

            # G feature loss
            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError(
                        'Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                self.netF = networks.define_F(
                    opt, use_bn=False).to(self.device)
                self.netF = DataParallel(self.netF)

            # GD gan loss
            self.cri_gan = GANLoss(
                train_opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan_w = train_opt['gan_weight']
            # D_update_ratio and D_init_iters
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger.warning(
                        'Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1_G'], train_opt['beta2_G']))
            self.optimizers.append(self.optimizer_G)
            # D
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'],
                                                weight_decay=wd_D,
                                                betas=(train_opt['beta1_D'], train_opt['beta2_D']))
            self.optimizers.append(self.optimizer_D)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError(
                    'MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        self.print_network()  # print network
        self.load()  # load G and D if needed

    def attack_fgsm(self, is_collect_data=False):
        # collect_data='collect_data' in self.opt['attack'] and self.opt['attack']['collect_data']

        for p in self.netD.parameters():
            p.requires_grad = False
        for p in self.netG.parameters():
            p.requires_grad = False
        self.var_L.requires_grad_()

        self.fake_H = self.netG(self.var_L)

        # l_g_total, l_g_pix, l_g_fea, l_g_gan=self.loss_for_G(self.fake_H,self.var_H,self.var_ref)
        l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)

        # zero_grad
        if self.var_L.grad is not None:
            self.var_L.grad.zero_()
        # self.netG.zero_grad()

        # l_g_total.backward()
        l_g_pix.backward()

        data_grad = self.var_L.grad.data

        sign_data_grad = data_grad.sign()
        perturbed_data = self.var_L + self.opt['attack']['eps']*sign_data_grad
        perturbed_data = torch.clamp(perturbed_data, 0, 1)

        if is_collect_data:
            init_data = self.var_L.detach()
            self.var_L = perturbed_data.detach()
            perturbed_data = self.var_L.clone().detach()
            return init_data, perturbed_data
        else:
            self.var_L = perturbed_data.detach()
            return

    # TODO test
    def attack_pgd(self, is_collect_data=False):
        eps = self.opt['attack']['eps']

        for p in self.netG.parameters():
            p.requires_grad = False
        orig_input = self.var_L.clone().detach()

        randn = torch.FloatTensor(self.var_L.size()).uniform_(-eps, eps).cuda()
        self.var_L += randn
        self.var_L.clamp_(0, 1.0)

        # self.var_L.requires_grad_()
        # if self.var_L.grad is not None:
        #     self.var_L.grad.zero_()
        self.var_L.detach_()

        for _ in range(self.opt['attack']['step_num']):
            # if self.var_L.grad is not None:
            #     self.var_L.grad.zero_()
            var_L_step = torch.autograd.Variable(
                self.var_L, requires_grad=True)
            self.fake_H = self.netG(var_L_step)
            l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
            l_pix.backward()
            data_grad = var_L_step.grad.data

            pert = self.opt['attack']['step']*data_grad.sign()
            self.var_L = self.var_L + pert.data
            self.var_L = torch.max(orig_input-eps, self.var_L)
            self.var_L = torch.min(orig_input+eps, self.var_L)
            self.var_L.clamp_(0, 1.0)

        if is_collect_data:
            return orig_input, self.var_L.clone().detach()
        else:
            self.var_L.detach_()
            return

    def feed_data(self, data, need_GT=True, is_collect_data=False):
        self.var_L = data['LQ'].to(self.device)  # LQ
        if need_GT:
            self.var_H = data['GT'].to(self.device)  # GT
            input_ref = data['ref'] if 'ref' in data else data['GT']
            self.var_ref = input_ref.to(self.device)

        # TODO attack code start
        if 'attack' in self.opt and need_GT and not ('raw_data' in self.opt['attack'] and self.opt['attack']['raw_data'] == True):
            if 'type' in self.opt['attack'] and self.opt['attack']['type'] == 'pgd':
                if not is_collect_data:
                    self.attack_pgd()
                else:
                    return self.attack_pgd(is_collect_data=True)
            else:
                if not is_collect_data:
                    self.attack_fgsm()
                else:
                    return self.attack_fgsm(is_collect_data=True)
        # attack code end

    def loss_for_G(self, fake_H, var_H, var_ref):
        l_g_total = 0
        if self.cri_pix:  # pixel loss
            l_g_pix = self.l_pix_w * self.cri_pix(fake_H, var_H)
            l_g_total += l_g_pix
        if self.cri_fea:  # feature loss
            real_fea = self.netF(var_H).detach()
            fake_fea = self.netF(fake_H)
            l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
            l_g_total += l_g_fea
        if self.l_gan_w > 0.0:
            if ('train' in self.opt and self.opt['train']['gan_type'] == 'gan') or ('attack' in self.opt and self.opt['gan_type'] == 'gan'):
                pred_g_fake = self.netD(fake_H)
                l_g_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
            elif ('train' in self.opt and self.opt['train']['gan_type'] == 'ragan') or ('attack' in self.opt and self.opt['gan_type'] == 'ragan'):
                pred_d_real = self.netD(var_ref).detach()
                pred_g_fake = self.netD(fake_H)
                l_g_gan = self.l_gan_w * (
                    self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                    self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
            l_g_total += l_g_gan
        else:
            l_g_gan = torch.tensor(0.0)
        return l_g_total, l_g_pix, l_g_fea, l_g_gan

    def optimize_parameters(self, step):
        # G
        for p in self.netD.parameters():
            p.requires_grad = False
        for p in self.netG.parameters():
            p.requires_grad = True
        if 'adv_train' in self.opt:
            self.var_L.requires_grad_()
            if self.var_L.grad is not None:
                self.var_L.grad.data.zero_()

        if 'adv_train' not in self.opt:
            self.fake_H = self.netG(self.var_L)
        else:
            self.fake_H = self.netG(torch.clamp(self.var_L+self.delta, 0, 1))

        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if 'adv_train' not in self.opt:
                l_g_total, l_g_pix, l_g_fea, l_g_gan = self.loss_for_G(
                    self.fake_H, self.var_H, self.var_ref)

                self.optimizer_G.zero_grad()

                l_g_total.backward()
                self.optimizer_G.step()
            else:
                for _ in range(self.opt['adv_train']['m']):
                    l_g_total, l_g_pix, l_g_fea, l_g_gan = self.loss_for_G(
                        self.fake_H, self.var_H, self.var_ref)

                    self.optimizer_G.zero_grad()
                    if self.var_L.grad is not None:
                        self.var_L.grad.data.zero_()

                    l_g_total.backward()
                    self.optimizer_G.step()

                    self.delta = self.delta + \
                        self.opt['adv_train']['step'] * \
                        self.var_L.grad.data.sign()
                    self.delta.clamp_(-self.opt['attack']
                                      ['eps'], self.opt['attack']['eps'])
                    self.fake_H = self.netG(
                        torch.clamp(self.var_L+self.delta, 0, 1))
        # D
        for p in self.netD.parameters():
            p.requires_grad = True

        self.optimizer_D.zero_grad()
        if self.opt['train']['gan_type'] == 'gan':
            # need to forward and backward separately, since batch norm statistics differ
            # real
            pred_d_real = self.netD(self.var_ref)
            l_d_real = self.cri_gan(pred_d_real, True)
            l_d_real.backward()
            # fake
            # detach to avoid BP to G
            pred_d_fake = self.netD(self.fake_H.detach())
            l_d_fake = self.cri_gan(pred_d_fake, False)
            l_d_fake.backward()
        elif self.opt['train']['gan_type'] == 'ragan':
            pred_d_fake = self.netD(self.fake_H.detach()).detach()
            pred_d_real = self.netD(self.var_ref)
            l_d_real = self.cri_gan(
                pred_d_real - torch.mean(pred_d_fake), True) * 0.5
            l_d_real.backward()
            pred_d_fake = self.netD(self.fake_H.detach())
            l_d_fake = self.cri_gan(
                pred_d_fake - torch.mean(pred_d_real.detach()), False) * 0.5
            l_d_fake.backward()
        self.optimizer_D.step()

        # set log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            self.log_dict['l_g_total'] = l_g_total.item()
            if self.cri_pix:
                self.log_dict['l_g_pix'] = l_g_pix.item()
            if self.cri_fea:
                self.log_dict['l_g_fea'] = l_g_fea.item()
            self.log_dict['l_g_gan'] = l_g_gan.item()

            self.log_dict['l_d_real'] = l_d_real.item()
            self.log_dict['l_d_fake'] = l_d_fake.item()
            self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
            self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.var_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)
        if self.is_train:
            # Discriminator
            s, n = self.get_network_description(self.netD)
            if isinstance(self.netD, nn.DataParallel):
                net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                 self.netD.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netD.__class__.__name__)
            logger.info('Network D structure: {}, with parameters: {:,d}'.format(
                net_struc_str, n))
            logger.info(s)

            if self.cri_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel):
                    net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                     self.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netF.__class__.__name__)
                logger.info('Network F structure: {}, with parameters: {:,d}'.format(
                    net_struc_str, n))
                logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG,
                              self.opt['path']['strict_load'])
        load_path_D = self.opt['path']['pretrain_model_D']
        if load_path_D is not None:
            logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD,
                              self.opt['path']['strict_load'])
        load_path_F = self.opt['path']['pretrain_model_F']
        if load_path_F is not None:
            logger.info('Loading model for F [{:s}] ...'.format(load_path_F))
            network = self.netF.module.features
            if isinstance(network, nn.DataParallel):
                network = network.module
            load_net = torch.load(load_path_F)
            load_net_clean = OrderedDict()  # remove unnecessary 'module.'
            for k, v in load_net.items():
                if k.startswith('module.features.'):
                    load_net_clean[k[16:]] = v
            network.load_state_dict(
                load_net_clean, strict=self.opt['path']['strict_load'])

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        self.save_network(self.netD, 'D', iter_step)
