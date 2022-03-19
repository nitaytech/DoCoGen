import logging
import random
from pathlib import Path
from typing import Any, Dict
from typing import List, Union

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer, LightningModule, Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import LightningLoggerBase
from project_utils.constants import PRECISION


class LoggingCallback(Callback):
    def __init__(self, output_dir: str,
                 logger=None,
                 save_to_csv: bool = False,
                 save_to_csv_every_n_step: int = 50):
        self.output_dir = output_dir
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
        self.save_to_csv = save_to_csv
        self.save_to_csv_every_n_step = save_to_csv_every_n_step
        self.current_step = 0

        self.train_epoch_output_results = []
        self.train_result_df = pd.DataFrame()
        self.metrics_df = pd.DataFrame()

        self.logger.info(f"Using GPU: {torch.cuda.is_available()}")

    def log_s(self, s: str):
        self.logger.info(s)

    def on_train_batch_end(self, trainer, pl_module: LightningModule, outputs: Any,
                           batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        if self.save_to_csv:
            results = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in outputs[-1][-1]['extra'].items()}
            results['loss'] = outputs[-1][-1]['minimize'].item()
            results['step'] = self.current_step
            results['step_group'] = self.current_step // self.save_to_csv_every_n_step
            results['epoch'] = pl_module.current_epoch
            self.train_epoch_output_results.append(results)
            if self.current_step % self.save_to_csv_every_n_step == 0:
                output_dir = Path(self.output_dir)
                file_path = output_dir / 'train_results.csv'
                df = pd.DataFrame(self.train_epoch_output_results).groupby(['step_group', 'epoch']).mean().reset_index()
                df = pd.concat([self.train_result_df, df], axis=0)
                df.to_csv(file_path, index=False)
        self.current_step += 1

    def on_train_end(self, trainer, pl_module: LightningModule) -> None:
        df = pd.DataFrame(self.train_epoch_output_results).groupby(['step_group', 'epoch']).mean().reset_index()
        self.train_result_df = pd.concat([self.train_result_df, df], axis=0).reset_index(drop=True)
        self.train_epoch_output_results = []
        if self.save_to_csv:
            output_dir = Path(self.output_dir)
            file_path = output_dir / 'train_results.csv'
            self.train_result_df.to_csv(file_path, index=False)

    def on_validation_end(self, trainer, pl_module):
        metrics = {k: v.item() for k, v in trainer.callback_metrics.items() if k not in ["log", "progress_bar"]}
        metrics['epoch'] = pl_module.current_epoch
        self.metrics_df = pd.concat([self.metrics_df, pd.DataFrame([metrics])], axis=0).reset_index(drop=True)
        if self.save_to_csv:
            output_dir = Path(self.output_dir)
            file_path = output_dir / 'metrics.csv'
            self.metrics_df.to_csv(file_path, index=False)

        self.logger.info("***** Validation results *****")
        # Log results
        for key, metric in metrics.items():
            self.log_s(f"{key} = {metric:.03f}")

    def on_test_end(self, trainer, pl_module):
        self.log_s("***** Test results *****")
        self.log_s(f"Num Training Epochs: {trainer.max_epochs}")
        # Log and save results to file
        output_dir = Path(self.output_dir)
        metrics = filter(lambda x: x[0] not in ["log", "progress_bar"], trainer.callback_metrics.items())
        with open(output_dir / "test_results.txt", "w") as writer:
            for key, metric in sorted(metrics):
                self.log_s(f"{key} = {metric:.03f}")
                writer.write(f"{key} = {metric:.05f}\n")


class ModelCheckpointWithResults(ModelCheckpoint):
    def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule) -> Dict[str, Any]:
        try:
            pl_module.write_outputs('checkpoint', to_csv=True)
        except Exception as e:
            pass
        return super().on_save_checkpoint(trainer, pl_module)


class T5Trainer(Trainer):
    def __init__(self,
                 output_dir: str,
                 accumulate_grad_batches: int = 1,
                 amp_level: str = 'O1',
                 callbacks: Union[List[Callback], Callback, None] = None,
                 fast_dev_run: bool = False,
                 gpus: int = 1,
                 gradient_clip_val: float = 1.0,
                 log_every_n_steps: int = 50,
                 logger: Union[LightningLoggerBase, bool] = False,
                 max_epochs: int = 100,
                 precision: int = PRECISION,
                 seed: int = None,
                 monitor: str = 'validation_mean_loss',
                 mode: str = 'min',
                 save_to_csv: bool = False,
                 save_to_csv_every_n_step: int = 50,
                 reload_dataloaders_every_epoch: bool = False
                 ):
        self.set_seed(seed)
        logging_callback = LoggingCallback(output_dir=output_dir, save_to_csv=save_to_csv,
                                           save_to_csv_every_n_step=save_to_csv_every_n_step)
        if callbacks is None:
            callbacks = []
        elif isinstance(callbacks, Callback):
            callbacks = [callbacks]
        checkpoint_callback = ModelCheckpointWithResults(dirpath=output_dir, filename=f'{monitor}_checkpoint',
                                                         monitor=monitor, mode=mode)
        callbacks.append(checkpoint_callback)
        if not fast_dev_run:
            callbacks = [logging_callback, checkpoint_callback]
        else:
            callbacks = [logging_callback]

        super().__init__(accumulate_grad_batches=accumulate_grad_batches, amp_level=amp_level, callbacks=callbacks,
                         default_root_dir=output_dir, fast_dev_run=fast_dev_run, gpus=gpus,
                         gradient_clip_val=gradient_clip_val, log_every_n_steps=log_every_n_steps, logger=logger,
                         max_epochs=max_epochs, precision=precision,
                         reload_dataloaders_every_epoch=reload_dataloaders_every_epoch)

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
