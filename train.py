import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from pytorch_lightning import loggers
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint

from model.model import get_model
from dataset.hubmap import get_dataset
from system.system import LitClassifier

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
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min')
    checkpoint_callback = ModelCheckpoint(dirpath=cfg.logging.log_dir,
                                          filename='fold' + str(cfg.dataset.target_fold)+'-{epoch}-{val_loss:.2f}',
                                          save_top_k=5, save_weights_only=True, mode='min', monitor='val_loss')

    # set data
    dataset = get_dataset(cfg.dataset)
    train_dataset = dataset['train']
    valid_dataset = dataset['valid']

    train_loader = DataLoader(train_dataset, **cfg.dataset.dataloader.train)
    val_loader = DataLoader(valid_dataset, **cfg.dataset.dataloader.valid)
    
    # set model
    model = get_model(cfg.model)

    # set lit system
    lit_model = LitClassifier(hparams=cfg, model=model)

    # set trainer
    trainer = Trainer(
        logger=[tb_logger],
        #callbacks=[lr_monitor, early_stopping, checkpoint_callback],
        callbacks=[lr_monitor, checkpoint_callback],
        **cfg.trainer
    )
    # training
    trainer.fit(lit_model, train_loader, val_loader)

    # test (if you need)
    #result = trainer.test(test_dataloaders=test_loader)
    #print(result)

if __name__ == "__main__":
    main()
