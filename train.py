import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from pytorch_lightning import loggers
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint

from model.model import get_model
from dataset.hubmap_ssl import get_dataset2
from system.system import LitClassifier
from callback.callback import ValidWholeImageCallback
from utils.utils import custom_load

# for temporary
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

class MyIter(object):
  """An iterator."""
  def __init__(self, my_loader):
    self.my_loader = my_loader
    self.loader_iters = [iter(loader) for loader in self.my_loader.loaders]

  def __iter__(self):
    return self

  def __next__(self):
    # When the shortest loader (the one with minimum number of batches)
    # terminates, this iterator will terminates.
    # The `StopIteration` raised inside that shortest loader's `__next__`
    # method will in turn gets out of this `__next__` method.
    batches = [loader_iter.next() for loader_iter in self.loader_iters]
    return self.my_loader.combine_batch(batches)

  # Python 2 compatibility
  next = __next__

  def __len__(self):
    return len(self.my_loader)

  
class MyLoader(object):
  """This class wraps several pytorch DataLoader objects, allowing each time 
  taking a batch from each of them and then combining these several batches 
  into one. This class mimics the `for batch in loader:` interface of 
  pytorch `DataLoader`.
  Args: 
    loaders: a list or tuple of pytorch DataLoader objects
  """
  def __init__(self, loaders):
    self.loaders = loaders

  def __iter__(self):
    return MyIter(self)

  def __len__(self):
    return min([len(loader) for loader in self.loaders])

  # Customize the behavior of combining batches here.
  def combine_batch(self, batches):
    return batches

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
    train_unlabeled = dataset['train_unlabel'] 
    valid_dataset = dataset['valid']

    train_labeled_loader = DataLoader(train_dataset, **cfg.dataset.dataloader.train)
    train_unlabeled_loader = DataLoader(train_unlabeled, **cfg.dataset.dataloader.train)
    train_loader = MyLoader([train_labeled_loader, train_unlabeled_loader])
    
    val_loader = DataLoader(valid_dataset, **cfg.dataset.dataloader.valid)
    #valid_callback = ValidWholeImageCallback(dataset['valid_img_id'], cfg.dataset, cfg.trainer.max_epochs/2, 10)
    # set model
    model = get_model(cfg.model)
    if cfg.model.ssl_model is not None:
        print('Loading ssl model: {}'.format(cfg.model.ssl_model))
        model.load_state_dict(custom_load(cfg.model.ssl_model), strict=False)

    # set lit system
    lit_model = LitClassifier(hparams=cfg, model=model)

    # set trainer
    trainer = Trainer(
        logger=[tb_logger],
        #callbacks=[lr_monitor, early_stopping, checkpoint_callback],
        #callbacks=[lr_monitor, checkpoint_callback, valid_callback],
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
