import click
from os.path import join, dirname, abspath
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import numpy as np
import torch
import yaml
import MinkowskiEngine as ME
import liflow.datasets 
import liflow.models
from liflow.utils.helpers import instantiate_from_config, load_model_from_config
from liflow.models.ema_callback import EMACallback
import warnings 

warnings.filterwarnings('ignore')

torch.set_float32_matmul_precision('medium')

def set_deterministic():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

@click.command()
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'config/config_flow.yaml'))
@click.option('--weights',
              '-w',
              type=str,
              help='path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.',
              default=None)
@click.option('--checkpoint',
              '-ckpt',
              type=str,
              help='path to checkpoint file (.ckpt) to resume training.',
              default=None)
@click.option('--test', '-t', is_flag=True, help='test mode')
def main(config, weights, checkpoint, test):
    set_deterministic()

    cfg = yaml.safe_load(open(config))
    

    #Load data and model
    if weights is None:
        model = instantiate_from_config(cfg['model'])
        model.save_hyperparameters(cfg)
    else:
        if test:
            ckpt_cfg = yaml.safe_load(open(weights.split('checkpoints')[0] + '/hparams.yaml'))
            ckpt_cfg['data']['params']['batch_size'] = cfg['data']['params']['batch_size']
            ckpt_cfg['data']['params']['num_workers'] = cfg['data']['params']['num_workers']
            ckpt_cfg['data']['params']['num_points'] = cfg['data']['params']['num_points']
            ckpt_cfg['data']['params']['data_dir'] = cfg['data']['params']['data_dir']
            ckpt_cfg['train']['n_gpus'] = cfg['train']['n_gpus']
            ckpt_cfg['experiment']['id'] = cfg['experiment']['id']


            cfg = ckpt_cfg

        model = load_model_from_config(cfg['model'], weights)
        model.save_hyperparameters(cfg)

    print(model.hparams)

    data = instantiate_from_config(cfg['data'])

    #Add callbacks
    callbacks = []

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    checkpoint_saver = ModelCheckpoint(
                                 save_last=True,
                                 save_top_k=0,
                                 )
    callbacks.append(checkpoint_saver)
    

    if cfg['train']['use_ema']:
        ema = EMACallback()
        callbacks.append(ema)
    
    log_dir = 'experiments/'+cfg['experiment']['id']

    tb_logger = pl_loggers.TensorBoardLogger(log_dir,
                                            default_hp_metric=False)
    

    #Setup trainer
    if torch.cuda.device_count() > 1:
        cfg['train']['n_gpus'] = torch.cuda.device_count()
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
        trainer = Trainer(devices=cfg['train']['n_gpus'],
                          logger=tb_logger,
                          log_every_n_steps=100,
                          max_epochs= cfg['train']['max_epoch'],
                          callbacks=callbacks,
                          check_val_every_n_epoch=5,
                          num_sanity_val_steps=0,
                          limit_val_batches=0.1,
                          accelerator='ddp',
                          )
    else:
        trainer = Trainer(devices=cfg['train']['n_gpus'],
                          logger=tb_logger,
                          log_every_n_steps=100,
                          max_epochs= cfg['train']['max_epoch'],
                          callbacks=callbacks,
                          check_val_every_n_epoch=5,
                          num_sanity_val_steps=0,
                          limit_val_batches=0.1,
                          accelerator='gpu',
                          )


    if test:
        print('TESTING MODE')
        trainer.test(model, data)
    else:
        print('TRAINING MODE')
        trainer.fit(model, data)

if __name__ == "__main__":
    main()
