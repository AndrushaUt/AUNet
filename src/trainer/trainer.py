from pathlib import Path

import pandas as pd

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_si_sdri
from src.trainer.base_trainer import BaseTrainer

import torch


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch_idx, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            if batch_idx % self.accumulation_steps == 0:
                self.optimizer.zero_grad()

        outputs = self.model(**batch)

        outputs = {
            "s1_estimated": outputs[:,0,:],
            "s2_estimated": outputs[:,1,:],
        }
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss

            if (batch_idx + 1) % self.accumulation_steps == 0:
                self._clip_grad_norm()
                self.optimizer.step()
                if self.lr_scheduler is not None and not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            if (batch_idx + 1) % self.accumulation_steps == 0:
                metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            if (batch_idx + 1) % self.accumulation_steps == 0:
                metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """

        self.log_audio(**batch)

    def log_audio(self, mix_audio, s1_audio, s2_audio, s1_estimated, s2_estimated, **batch):
        self.writer.add_audio("mix_audio", mix_audio[0], 16000)
        self.writer.add_audio("s1_audio", s1_audio[0], 16000)
        self.writer.add_audio("s1_estimated", s1_estimated[0], 16000)
        self.writer.add_audio("s2_audio", s2_audio[0], 16000)
        self.writer.add_audio("s2_estimated", s2_estimated[0], 16000)