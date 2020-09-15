from models import create_model
from data import create_dataloader, create_dataset
from utils import util
import options.options as option
import torch.multiprocessing as mp
import torch
import time
import logging
import random
import argparse
import math
import os
import faulthandler
faulthandler.enable()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, help='Path to option YAML file.')
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    #### loading resume state if exists
    if 'resume_latest' in opt and opt['resume_latest'] == True:
        if os.path.isdir(opt['path']['training_state']):
            name_state_files = os.listdir(opt['path']['training_state'])
            if len(name_state_files) > 0:
                latest_state_num = 0
                for name_state_file in name_state_files:
                    state_num = int(name_state_file.split('.')[0])
                    if state_num > latest_state_num:
                        latest_state_num = state_num
                opt['path']['resume_state'] = os.path.join(
                    opt['path']['training_state'], str(latest_state_num)+'.state')
            else:
                raise ValueError
    if opt['path'].get('resume_state', None):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if resume_state is None:
        util.mkdir_and_rename(
            opt['path']['experiments_root'])  # rename experiment folder if exists
        util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                     and 'pretrain_model' not in key and 'resume' not in key))

    # config loggers. Before it, the log will not work
    util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                      screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))
    # tensorboard logger
    if opt['use_tb_logger'] and 'debug' not in opt['name']:
        version = float(torch.__version__[0:3])
        if version >= 1.1:  # PyTorch 1.1
            from torch.utils.tensorboard import SummaryWriter
        else:
            logger.info(
                'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
            from tensorboardX import SummaryWriter
        tb_logger = SummaryWriter(log_dir='../tb_logger/' +
                                  opt['name'] + '_{}'.format(util.get_timestamp()))

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(
                math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            train_sampler = None
            train_loader = create_dataloader(
                train_set, dataset_opt, opt, train_sampler)
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                len(train_set), train_size))
            logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            logger.info('Number of val images in [{:s}]: {:d}'.format(
                dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError(
                'Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    is_time = False

    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(
        start_epoch, current_step))

    if is_time:
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')

    for epoch in range(start_epoch, total_epochs + 1):
        if current_step > total_iters:
            break

        if is_time:
            torch.cuda.synchronize()
            end = time.time()
        for _, train_data in enumerate(train_loader):
            if 'adv_train' in opt:
                current_step += opt['adv_train']['m']
            else:
                current_step += 1
            if current_step > total_iters:
                break

            #### training
            model.feed_data(train_data)

            if is_time:
                torch.cuda.synchronize()
                data_time.update(time.time() - end)

            model.optimize_parameters(current_step)

            #### update learning rate
            model.update_learning_rate(
                current_step, warmup_iter=opt['train']['warmup_iter'])

            if is_time:
                torch.cuda.synchronize()
                batch_time.update(time.time() - end)

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                # FIXME remove debug
                debug = True
                if debug:
                    torch.cuda.empty_cache()
                logs = model.get_current_log()
                message = '[epoch:{:3d}, iter:{:8,d}, lr:('.format(
                    epoch, current_step)
                for v in model.get_current_learning_rate():
                    message += '{:.3e},'.format(v)
                message += ')] '
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        tb_logger.add_scalar(k, v, current_step)
                logger.info(message)
                if is_time:
                    logger.info(str(data_time))
                    logger.info(str(batch_time))

            #### validation
            if opt['datasets'].get('val', None) and current_step % opt['train']['val_freq'] == 0:
                if opt['model'] in ['sr', 'srgan']:  # image restoration validation
                    pbar = util.ProgressBar(len(val_loader))
                    avg_psnr = 0.
                    idx = 0
                    for val_data in val_loader:
                        idx += 1
                        img_name = os.path.splitext(
                            os.path.basename(val_data['LQ_path'][0]))[0]
                        img_dir = os.path.join(
                            opt['path']['val_images'], img_name)
                        util.mkdir(img_dir)

                        model.feed_data(val_data)
                        model.test()

                        visuals = model.get_current_visuals()
                        sr_img = util.tensor2img(visuals['rlt'])  # uint8
                        gt_img = util.tensor2img(visuals['GT'])  # uint8

                        # Save SR images for reference
                        save_img_path = os.path.join(img_dir,
                                                     '{:s}_{:d}.png'.format(img_name, current_step))
                        util.save_img(sr_img, save_img_path)

                        # calculate PSNR
                        sr_img, gt_img = util.crop_border(
                            [sr_img, gt_img], opt['scale'])
                        avg_psnr += util.calculate_psnr(sr_img, gt_img)
                        pbar.update('Test {}'.format(img_name))

                    avg_psnr = avg_psnr / idx

                    # log
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        tb_logger.add_scalar('psnr', avg_psnr, current_step)

            #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(current_step)
                model.save_training_state(epoch, current_step)
                tb_logger.flush()
            if is_time:
                torch.cuda.synchronize()
                end = time.time()

    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')
    tb_logger.close()


if __name__ == '__main__':
    main()
