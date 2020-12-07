import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from pytorch_lightning import loggers
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint

from model.model import get_model
from dataset.hubmap import get_dataset2
from system.system import LitClassifier
from callback.callback import ValidWholeImageCallback
# for temporary
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

@hydra.main(config_path='config', config_name="config.yaml")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # set seed
    seed_everything(2020)

    # set logger
    tb_logger = loggers.TensorBoardLogger(**cfg.logging.tb_logger)
    # set callback
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stopping = EarlyStopping(monitor='val_dice', patience=20, mode='min')
    checkpoint_callback = ModelCheckpoint(dirpath=cfg.logging.log_dir,
                                          filename='fold' + str(cfg.dataset.target_fold)+'-{epoch}-{val_dice:.5f}',
                                          save_top_k=5, save_weights_only=True, mode='max', monitor='val_dice')

    # set data
    dataset = get_dataset2(cfg.dataset)
    train_dataset = dataset['train']
    valid_dataset = dataset['valid']

    train_loader = DataLoader(train_dataset, **cfg.dataset.dataloader.train)
    val_loader = DataLoader(valid_dataset, **cfg.dataset.dataloader.valid)
    valid_callback = ValidWholeImageCallback(dataset['valid_img_id'], cfg.dataset, cfg.trainer.max_epochs/2, 10)
    # set model
    model = get_model(cfg.model)

    # set lit system
    lit_model = LitClassifier(hparams=cfg, model=model)

    # set trainer
    trainer = Trainer(
        logger=[tb_logger],
        #callbacks=[lr_monitor, early_stopping, checkpoint_callback],
        callbacks=[lr_monitor, checkpoint_callback, valid_callback],
        **cfg.trainer
    )
    # training
    trainer.fit(lit_model, train_loader, val_loader)

    # test (if you need)
    #result = trainer.test(test_dataloaders=test_loader)
    #print(result)

if __name__ == "__main__":
    main()
