import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss

logger = logging.getLogger('base')


class SRModel(BaseModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if not self.is_train and 'attack' in self.opt:
            # loss
            loss_type = opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            else:
                raise NotImplementedError(
                    'Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = opt['pixel_weight']

        # FIXME
        # input_size=opt['datasets']['train']['GT_size']//opt['datasets']['train']['scale']
        # self.delta=torch.zeros(1,3,input_size,input_size).cuda()

        self.delta = 0

        if self.is_train:
            self.netG.train()

            # loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            else:
                raise NotImplementedError(
                    'Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            # optimizers
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
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

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

    def attack_fgsm(self, is_collect_data=False):
        # collect_data='collect_data' in self.opt['attack'] and self.opt['attack']['collect_data']

        for p in self.netG.parameters():
            p.requires_grad = False
        self.var_L.requires_grad_()

        self.fake_H = self.netG(self.var_L)
        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)

        if self.var_L.grad is not None:
            self.var_L.grad.zero_()
        # self.netG.zero_grad()

        l_pix.backward()
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

    # TODO  test
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
            l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
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
            self.real_H = data['GT'].to(self.device)  # GT
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

    def optimize_parameters(self, step):
        for p in self.netG.parameters():
            p.requires_grad = True
        if 'adv_train' not in self.opt:
            self.optimizer_G.zero_grad()
            self.fake_H = self.netG(self.var_L)
            l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
            l_pix.backward()
            self.optimizer_G.step()
        else:
            self.var_L.requires_grad_()
            for _ in range(self.opt['adv_train']['m']):
                self.optimizer_G.zero_grad()
                if self.var_L.grad is not None:
                    self.var_L.grad.data.zero_()

                self.fake_H = self.netG(
                    torch.clamp(self.var_L+self.delta, 0, 1))
                l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
                l_pix.backward()
                self.optimizer_G.step()

                self.delta = self.delta + \
                    self.opt['adv_train']['step']*self.var_L.grad.data.sign()
                self.delta.clamp_(-self.opt['attack']
                                  ['eps'], self.opt['attack']['eps'])

        # set log
        self.log_dict['l_pix'] = l_pix.item()

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
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG,
                              self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
