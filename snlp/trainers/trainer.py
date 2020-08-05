#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: trainer.py
@time: 2019/12/3 19:23
@description: 通用的模型训练类
'''

import typing
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter

import snlp.tasks as tasks
from snlp.base import BaseModel, BaseMetric
from snlp.tools.common import parse_device
from snlp.tools.log import logger
from snlp.tools.average_meter import AverageMeter
from snlp.tools.timer import Timer
from snlp.tools.early_stopping import EarlyStopping

logger = logger.getChild("-trainer")

class Trainer:
    """
    Generic Trainer.
    下面参数中的epoch_scheduler和step_scheduler每次只能指定一个
    """
    def __init__(
        self,
        model: BaseModel,
        optimizer: Optimizer,
        trainloader: DataLoader,
        validloader: DataLoader,
        device: typing.Union[torch.device, int, list, None] = None,
        writer: SummaryWriter=None,
        start_epoch: int = 1,
        epochs: int = 10,
        validate_interval: typing.Optional[int] = None,
        epoch_scheduler: typing.Any = None,
        step_scheduler: typing.Any = None,
        clip_norm: typing.Union[float, int] = None,
        l1_reg: float = 0.0,
        l2_reg: float = 0.0,
        patience: typing.Optional[int] = None,
        key: typing.Any = None,
        checkpoint: typing.Union[str, Path] = None,
        save_dir: typing.Union[str, Path] = None,
        save_all: bool = False,
        verbose: int = 1,
        **kwargs
    ):
        log_dir = Path(save_dir).joinpath("runs")
        self._writer = writer or SummaryWriter(log_dir=log_dir)
        logger.info(f" ** save tensorboard to {self._writer.get_logdir()} ** ")

        self._load_model(model, device)
        self._load_dataloader(
            trainloader, validloader, validate_interval
        )
        self._optimizer = optimizer
        self._epoch_scheduler = epoch_scheduler   # 由于这里的scheduler是在每个epoch之后调用一次，所以在定义的时候注意设置更新的step数和epoch一致
        self._step_scheduler = step_scheduler  # 这个是针对step的scheduler
        self._clip_norm = clip_norm # 梯度裁剪
        # 正则化系数
        self._l1_reg = l1_reg
        self._l2_reg = l2_reg
        self._criterions = self._task.losses

        if not key:
            key = self._task.metrics[0]
        self._early_stopping = EarlyStopping(
            patience=patience,
            key=key
        )

        self._start_epoch = start_epoch
        self._epochs = epochs
        self._iteration = 0
        self._verbose = verbose
        self._save_all = save_all

        self._load_path(checkpoint, save_dir)

    def _load_model(self,
                    model: BaseModel,
                    device: typing.Union[torch.device, int, list, None]=None):
        if not isinstance(model, BaseModel):
            raise ValueError(
                f"model should be a `BaseModel` instance. "
                f"But got {type(model)}"
            )

        logger.info(" ** load model and device ** ")
        self._task = model.params['task']
        self._data_parallel = False
        self._model = model

        # 如果指定了多个GPU，则默认是数据并行
        if isinstance(device, list) and len(device):
            logger.info(" ** data parallel ** ")
            self._data_parallel = True
            self._model = torch.nn.DataParallel(self._model, device_ids=device)
            self._device = device[0]
        else:
            self._device = parse_device(device)
        self._model.to(self._device)

    def _load_dataloader(self,
                         trainloader: DataLoader,
                         validloader: DataLoader,
                         validate_interval: typing.Optional[int] = None):
        if not isinstance(trainloader, DataLoader):
            raise ValueError(
                'trainloader should be a `DataLoader` instance.'
            )
        if not isinstance(validloader, DataLoader):
            raise ValueError(
                'validloader should be a `DataLoader` instance.'
            )
        self._trainloader = trainloader
        self._validloader = validloader
        if not validate_interval:
            self._validate_interval = len(self._trainloader)
        else:
            self._validate_interval = validate_interval


    def _load_path(self,
                   checkpoint: typing.Union[str, Path],
                   save_dir: typing.Union[str, Path]):
        logger.info(" ** restore exist model checkpoint ** ")
        if not save_dir:
            save_dir = Path('.').joinpath('save')
            if not Path(save_dir).exists():
                Path(save_dir).mkdir(parents=True)


        self._save_dir = Path(save_dir)

        if checkpoint:
            if self._save_all:
                self.restore(checkpoint)
            else:
                self.restore_model(checkpoint)

    def _backward(self, loss):
        """
        Computes the gradient of current `loss` graph leaves.

        :param loss: Tensor. Loss of model.

        """
        # 先将梯度置为0
        self._optimizer.zero_grad()
        # 计算参数正则化的值
        if self._l1_reg > 0.0 or self._l2_reg > 0.0:
            l1_regularization = torch.tensor(0, dtype=torch.float, device=self._device)
            l2_regularization = torch.tensor(0, dtype=torch.float, device=self._device)
            for name, p in self._model.named_parameters():
                if "bias" not in name and "embedding" not in name:
                    if self._l1_reg > 0.0:
                        l1_regularization += torch.norm(p, 1)
                    if self._l2_reg > 0.0:
                        l2_regularization += torch.norm(p, 2)
            loss = loss + self._l1_reg * l1_regularization + self._l2_reg * l2_regularization

        loss.backward()
        if self._clip_norm:
            nn.utils.clip_grad_norm_(
                self._model.parameters(), self._clip_norm
            )
        self._optimizer.step()
        if self._step_scheduler:
            self._step_scheduler.step()


    def _run_scheduler(self):
        if self._epoch_scheduler:
            self._epoch_scheduler.step()

    def run(self):
        """
        Train model.

        Run each epoch -> Run scheduler -> Should stop early?
        """
        self._model.train()
        timer = Timer()
        for epoch in range(self._start_epoch, self._epochs + 1):
            self._epoch = epoch
            self._run_epoch()
            self._run_scheduler()  # 每个epoch对学习率更新一次
            if self._early_stopping.should_stop_early:
                break
        if self._verbose:
            tqdm.write(f"Cost time: {timer.time} s")

    def _run_epoch(self):
        """
        Run each epoch.

        The training steps:
            - Get batch and feed them into model
            - Get outputs. Caculate all losses and sum them up
            - Loss backwards and optimizer steps
            - Evaluation
            - Update and output result
        """
        num_batch = len(self._trainloader)
        train_loss = AverageMeter()
        with tqdm(enumerate(self._trainloader), total=num_batch,
                  disable=not self._verbose) as pbar:
            for step, (inputs, target) in pbar:
                # Run Train
                outputs = self._model(inputs)
                ## 计算所有的loss并相加
                loss = sum(
                    [c(outputs, target) for c in self._criterions]
                )

                self._backward(loss)
                ## 更新loss的值
                train_loss.update(loss.item())

                ## 设置Progress Bar
                pbar.set_description(f"Epoch {self._epoch}/{self._epochs}")
                pbar.set_postfix(loss=f"{loss.item():.3f}")
                self._writer.add_scalar("Loss/train", loss.item(), self._iteration)

                # Run Evaluate
                self._iteration += 1
                if self._iteration % self._validate_interval == 0:
                    pbar.update(1)
                    if self._verbose:
                        pbar.write(
                            f"[Iter-{self._iteration} "
                            f"Loss-{train_loss.avg:.3f}]")
                    ## 更新validloader评估的结果
                    result, eval_loss = self.evaluate(self._validloader)
                    m_string = ""
                    for metric in result:
                        res = result[metric]
                        if not isinstance(metric, str):
                            metric = metric.ALIAS[0]
                        self._writer.add_scalar(f"{metric}/eval", res, self._iteration)
                        m_string += f"| {metric}: {res} "

                    logger.info(f"Epoch: {self._epoch} | Train Loss: {loss.item():.3f} | "
                                f"Eval loss: {eval_loss.item(): .3f} " + m_string)
                    if self._verbose:
                        pbar.write(" Validation: " + '-'.join(
                            f"{k}: {round(v, 4)}" for k, v in result.items()))
                    ## Early Stopping
                    self._early_stopping.update(result)
                    if self._early_stopping.should_stop_early:
                        self._save()
                        pbar.write("Ran out of patience. Stop training...")
                        break
                    elif self._early_stopping.is_best_so_far:
                        self._save()

    def evaluate(self,
                 dataloader: DataLoader):
        result = dict()
        y_pred = self.predict(dataloader)
        y_true = dataloader.label
        if isinstance(self._task, tasks.Classification):
            y_pred_label = np.argmax(y_pred[:10], axis=-1)
            # 记录前10个真实标签和预测标签
            logger.info(f"The former 10 true label is {y_true[:10]}  | pred label is {y_pred_label}")
        elif isinstance(self._task, tasks.Ranking):
            y_true = y_true.reshape(len(y_true), 1)
        loss = sum(
            [c(torch.tensor(y_pred), torch.tensor(y_true)) for c in self._criterions]
        )
        self._writer.add_scalar("Loss/eval", loss.item(), self._iteration)
        try:
            id_left = dataloader.id_left
        except:
            pass

        if isinstance(self._task, tasks.Ranking):
            for metric in self._task.metrics:
                result[metric] = self._eval_metric_on_data_frame(
                    metric, id_left, y_true, y_pred.squeeze(axis=-1)
                )
        else:
            for metric in self._task.metrics:
                result[metric] = metric(y_true, y_pred)
        return result, loss


    @classmethod
    def _eval_metric_on_data_frame(cls,
                                   metric: BaseMetric,
                                   id_left: typing.Any,
                                   y_true: typing.Union[list, np.ndarray],
                                   y_pred: typing.Union[list, np.ndarray]):
        eval_df = pd.DataFrame(data={
            'id': id_left,
            'true': y_true,
            'pred': y_pred
        })
        assert isinstance(metric, BaseMetric)
        val = eval_df.groupby(by='id').apply(
            lambda df: metric(df['true'].values, df['pred'].values)
        ).mean()
        return val

    def predict(self,
                dataloader: DataLoader) -> np.ndarray:
        with torch.no_grad():
            self._model.eval()
            predictions = []
            for batch in dataloader:
                inputs = batch[0]
                outputs = self._model(inputs).detach().cpu()
                predictions.append(outputs)
            self._model.train()
            return torch.cat(predictions, dim=0).numpy()

    def _save(self):
        if self._save_all:
            self.save()
        else:
            self.save_model()

    def save_model(self):
        """Save the model."""
        checkpoint = self._save_dir.joinpath('model.pt')
        logger.info(f" ** save raw model to {checkpoint} ** ")
        if self._data_parallel:
            torch.save(self._model.module.state_dict(), checkpoint)
        else:
            torch.save(self._model.state_dict(), checkpoint)

    def save(self):
        """
        Save the trainer.

        `Trainer` parameters like epoch, best_so_far, model, optimizer
        and early_stopping will be savad to specific file path.

        :param path: Path to save trainer.

        """
        checkpoint = self._save_dir.joinpath('trainer.pt')
        logger.info(f" ** save trainer model to {checkpoint} ** ")
        if self._data_parallel:
            model = self._model.module.state_dict()
        else:
            model = self._model.state_dict()
        state = {
            'epoch': self._epoch,
            'model': model,
            'optimizer': self._optimizer.state_dict(),
            'early_stopping': self._early_stopping.state_dict(),
        }
        if self._epoch_scheduler:
            state['epoch_scheduler'] = self._epoch_scheduler.state_dict()
        if self._step_scheduler:
            state['step_scheduler'] = self._step_scheduler.state_dict()
        torch.save(state, checkpoint)

    def restore_model(self, checkpoint: typing.Union[str, Path]):
        """
        Restore model.

        :param checkpoint: A checkpoint from which to continue training.

        """
        logger.info(" ** restore raw model ** ")
        state = torch.load(checkpoint, map_location=self._device)
        if self._data_parallel:
            self._model.module.load_state_dict(state)
        else:
            self._model.load_state_dict(state)

    def restore(self, checkpoint: typing.Union[str, Path] = None):
        """
        Restore trainer.

        :param checkpoint: A checkpoint from which to continue training.

        """
        logger.info(" ** restore trainer model ** ")
        state = torch.load(checkpoint, map_location=self._device)
        if self._data_parallel:
            self._model.module.load_state_dict(state['model'])
        else:
            self._model.load_state_dict(state['model'])
        self._optimizer.load_state_dict(state['optimizer'])
        self._start_epoch = state['epoch'] + 1
        self._early_stopping.load_state_dict(state['early_stopping'])
        if self._epoch_scheduler:
            self._epoch_scheduler.load_state_dict(state['epoch_scheduler'])
        if self._step_scheduler:
            self._step_scheduler.load_state_dict(state['step_scheduler'])
