import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from model import Model, Config, TextPreprocessing, MyDataset, get_collate_fn

if __name__=="__main__":
    pl.utilities.seed.seed_everything(0)
    config = Config()

    train_df = pd.read_csv("data/train.csv")
    data_preprocessing = TextPreprocessing(config.max_seq_len, train_df, "vocab.pth")

    train_dataset = MyDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             collate_fn=get_collate_fn(data_preprocessing.text2tensor), 
                             shuffle=True)

    model = Model(config)
    gpus = 1 if torch.cuda.is_available() else 0
    trainer = pl.Trainer(max_epochs=Config().max_epochs, gpus=gpus)
    trainer.fit(model, train_dataloaders=train_loader)