"""create dataset and dataloader"""
import logging
import torch
import torch.utils.data

from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__() ,max_prefetch=16)


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    # FIXME remove debug
    debug=True
    debug=False

    phase = dataset_opt['phase']
    if phase == 'train':
        num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
        batch_size = dataset_opt['batch_size']
        shuffle = True
        if debug:
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                            num_workers=0, sampler=sampler, drop_last=True,
                                            )
        else:
            return DataLoaderX(dataset, batch_size=batch_size, shuffle=shuffle,
                                            num_workers=num_workers, sampler=sampler, drop_last=True,
                                            )
    else:
        if debug:
            return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                                            )
        else:
            return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=3,
                                            )


def create_dataset(dataset_opt):
    mode = dataset_opt['mode']
    # datasets for image restoration
    if mode == 'LQ':
        from data.LQ_dataset import LQDataset as D
    elif mode == 'LQGT':
        from data.LQGT_dataset import LQGTDataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt)

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
