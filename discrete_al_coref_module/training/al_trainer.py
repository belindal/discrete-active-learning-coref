"""
A :class:`~allennlp.training.trainer.Trainer` is responsible for training a
:class:`~allennlp.models.model.Model`.

Typically you might create a configuration file specifying the model and
training parameters and then use :mod:`~allennlp.commands.train`
rather than instantiating a ``Trainer`` yourself.
"""
# pylint: disable=too-many-lines

import logging
import json
import os
import shutil
import time
import re
import datetime
import traceback
import numpy as np
import math
from typing import Dict, Optional, List, Tuple, Union, Iterable, Any, Set
from retrying import retry

import torch
import torch.optim.lr_scheduler
from torch.nn.parallel import replicate, parallel_apply
from torch.nn.parallel.scatter_gather import gather
from tensorboardX import SummaryWriter

from allennlp.common import Params, Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import dump_metrics, gpu_memory_mb, peak_memory_mb
from allennlp.common.tqdm import Tqdm
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.fields import SequenceLabelField, SequenceField, ListField, IndexField
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.metrics import MentionRecall, ConllCorefScores
from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer_base import TrainerBase

from discrete_al_coref_module.dataset_readers.pair_field import PairField
from discrete_al_coref_module.training import active_learning_coref_utils as al_util

import random

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


PAIRWISE_Q_TIME = 15.961803738317756
DISCRETE_Q_TIME_TOTAL = 15.573082474226803 + PAIRWISE_Q_TIME
DISCRETE_PAIRWISE_RATIO = DISCRETE_Q_TIME_TOTAL / PAIRWISE_Q_TIME


def is_sparse(tensor):
    return tensor.is_sparse


def sparse_clip_norm(parameters, max_norm, norm_type=2) -> float:
    """Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    Supports sparse gradients.

    Parameters
    ----------
    parameters : ``(Iterable[torch.Tensor])``
        An iterable of Tensors that will have gradients normalized.
    max_norm : ``float``
        The max norm of the gradients.
    norm_type : ``float``
        The type of the used p-norm. Can be ``'inf'`` for infinity norm.

    Returns
    -------
    Total norm of the parameters (viewed as a single vector).
    """
    # pylint: disable=invalid-name,protected-access
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            if is_sparse(p.grad):
                # need to coalesce the repeated indices before finding norm
                grad = p.grad.data.coalesce()
                param_norm = grad._values().norm(norm_type)
            else:
                param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if is_sparse(p.grad):
                p.grad.data._values().mul_(clip_coef)
            else:
                p.grad.data.mul_(clip_coef)
    return total_norm


def move_optimizer_to_cuda(optimizer):
    """
    Move the optimizer state to GPU, if necessary.
    After calling, any parameter specific state in the optimizer
    will be located on the same device as the parameter.
    """
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            if param.is_cuda:
                param_state = optimizer.state[param]
                for k in param_state.keys():
                    if isinstance(param_state[k], torch.Tensor):
                        param_state[k] = param_state[k].cuda(device=param.get_device())


class TensorboardWriter:
    """
    Wraps a pair of ``SummaryWriter`` instances but is a no-op if they're ``None``.
    Allows Tensorboard logging without always checking for Nones first.
    """
    def __init__(self, train_log: SummaryWriter = None, validation_log: SummaryWriter = None) -> None:
        self._train_log = train_log
        self._validation_log = validation_log

    @staticmethod
    def _item(value: Any):
        if hasattr(value, 'item'):
            val = value.item()
        else:
            val = value
        return val

    def add_train_scalar(self, name: str, value: float, global_step: int) -> None:
        # get the scalar
        if self._train_log is not None:
            self._train_log.add_scalar(name, self._item(value), global_step)

    def add_train_histogram(self, name: str, values: torch.Tensor, global_step: int) -> None:
        if self._train_log is not None:
            if isinstance(values, torch.Tensor):
                values_to_write = values.cpu().data.numpy().flatten()
                self._train_log.add_histogram(name, values_to_write, global_step)

    def add_validation_scalar(self, name: str, value: float, global_step: int) -> None:

        if self._validation_log is not None:
            self._validation_log.add_scalar(name, self._item(value), global_step)


def time_to_str(timestamp: int) -> str:
    """
    Convert seconds past Epoch to human readable string.
    """
    datetimestamp = datetime.datetime.fromtimestamp(timestamp)
    return '{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}'.format(
            datetimestamp.year, datetimestamp.month, datetimestamp.day,
            datetimestamp.hour, datetimestamp.minute, datetimestamp.second
    )


def str_to_time(time_str: str) -> datetime.datetime:
    """
    Convert human readable string to datetime.datetime.
    """
    pieces: Any = [int(piece) for piece in time_str.split('-')]
    return datetime.datetime(*pieces)


@TrainerBase.register("al_coref_trainer")
class ALCorefTrainer(TrainerBase):
    def __init__(self,
                 model: Model,
                 optimizer: torch.optim.Optimizer,
                 iterator: DataIterator,
                 train_dataset: Iterable[Instance],
                 held_out_train_dataset: Optional[Iterable[Instance]] = None,
                 validation_dataset: Optional[Iterable[Instance]] = None,
                 patience: Optional[int] = None,
                 validation_metric: str = "-loss",
                 validation_iterator: DataIterator = None,
                 held_out_iterator: DataIterator = None,
                 shuffle: bool = True,
                 num_epochs: int = 20,
                 serialization_dir: Optional[str] = None,
                 num_serialized_models_to_keep: int = 20,
                 keep_serialized_model_every_num_seconds: int = None,
                 model_save_interval: float = None,
                 cuda_device: Union[int, List] = -1,
                 grad_norm: Optional[float] = None,
                 grad_clipping: Optional[float] = None,
                 learning_rate_scheduler: Optional[LearningRateScheduler] = None,
                 summary_interval: int = 100,
                 histogram_interval: int = None,
                 should_log_parameter_statistics: bool = True,
                 should_log_learning_rate: bool = False,
                 active_learning: Optional[Dict[str, int]] = None,
                 ensemble_model: Optional[Model] = None,
                 ensemble_optimizer: Optional[List[Optimizer]] = None,
                 ensemble_scheduler: Optional[List[LearningRateScheduler]] = None) -> None:
        """
        Parameters
        ----------
        model : ``Model``, required.
            An AllenNLP model to be optimized. Pytorch Modules can also be optimized if
            their ``forward`` method returns a dictionary with a "loss" key, containing a
            scalar tensor representing the loss function to be optimized.
        optimizer : ``torch.nn.Optimizer``, required.
            An instance of a Pytorch Optimizer, instantiated with the parameters of the
            model to be optimized.
        iterator : ``DataIterator``, required.
            A method for iterating over a ``Dataset``, yielding padded indexed batches.
        train_dataset : ``Dataset``, required.
            A ``Dataset`` to train on. The dataset should have already been indexed.
        validation_dataset : ``Dataset``, optional, (default = None).
            A ``Dataset`` to evaluate on. The dataset should have already been indexed.
        patience : Optional[int] > 0, optional (default=None)
            Number of epochs to be patient before early stopping: the training is stopped
            after ``patience`` epochs with no improvement. If given, it must be ``> 0``.
            If None, early stopping is disabled.
        validation_metric : str, optional (default="loss")
            Validation metric to measure for whether to stop training using patience
            and whether to serialize an ``is_best`` model each epoch. The metric name
            must be prepended with either "+" or "-", which specifies whether the metric
            is an increasing or decreasing function.
        validation_iterator : ``DataIterator``, optional (default=None)
            An iterator to use for the validation set.  If ``None``, then
            use the training `iterator`.
        shuffle: ``bool``, optional (default=True)
            Whether to shuffle the instances in the iterator or not.
        num_epochs : int, optional (default = 20)
            Number of training epochs.
        serialization_dir : str, optional (default=None)
            Path to directory for saving and loading model files. Models will not be saved if
            this parameter is not passed.
        num_serialized_models_to_keep : ``int``, optional (default=20)
            Number of previous model checkpoints to retain.  Default is to keep 20 checkpoints.
            A value of None or -1 means all checkpoints will be kept.
        keep_serialized_model_every_num_seconds : ``int``, optional (default=None)
            If num_serialized_models_to_keep is not None, then occasionally it's useful to
            save models at a given interval in addition to the last num_serialized_models_to_keep.
            To do so, specify keep_serialized_model_every_num_seconds as the number of seconds
            between permanently saved checkpoints.  Note that this option is only used if
            num_serialized_models_to_keep is not None, otherwise all checkpoints are kept.
        model_save_interval : ``float``, optional (default=None)
            If provided, then serialize models every ``model_save_interval``
            seconds within single epochs.  In all cases, models are also saved
            at the end of every epoch if ``serialization_dir`` is provided.
        cuda_device : ``int``, optional (default = -1)
            An integer specifying the CUDA device to use. If -1, the CPU is used.
        grad_norm : ``float``, optional, (default = None).
            If provided, gradient norms will be rescaled to have a maximum of this value.
        grad_clipping : ``float``, optional (default = ``None``).
            If provided, gradients will be clipped `during the backward pass` to have an (absolute)
            maximum of this value.  If you are getting ``NaNs`` in your gradients during training
            that are not solved by using ``grad_norm``, you may need this.
        learning_rate_scheduler : ``PytorchLRScheduler``, optional, (default = None)
            A Pytorch learning rate scheduler. The learning rate will be decayed with respect to
            this schedule at the end of each epoch. If you use
            :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`, this will use the ``validation_metric``
            provided to determine if learning has plateaued.  To support updating the learning
            rate on every batch, this can optionally implement ``step_batch(batch_num_total)`` which
            updates the learning rate given the batch number.
        summary_interval: ``int``, optional, (default = 100)
            Number of batches between logging scalars to tensorboard
        histogram_interval : ``int``, optional, (default = ``None``)
            If not None, then log histograms to tensorboard every ``histogram_interval`` batches.
            When this parameter is specified, the following additional logging is enabled:
                * Histograms of model parameters
                * The ratio of parameter update norm to parameter norm
                * Histogram of layer activations
            We log histograms of the parameters returned by
            ``model.get_parameters_for_histogram_tensorboard_logging``.
            The layer activations are logged for any modules in the ``Model`` that have
            the attribute ``should_log_activations`` set to ``True``.  Logging
            histograms requires a number of GPU-CPU copies during training and is typically
            slow, so we recommend logging histograms relatively infrequently.
            Note: only Modules that return tensors, tuples of tensors or dicts
            with tensors as values currently support activation logging.
        should_log_parameter_statistics : ``bool``, optional, (default = True)
            Whether to send parameter statistics (mean and standard deviation
            of parameters and gradients) to tensorboard.
        should_log_learning_rate : ``bool``, optional, (default = False)
            Whether to send parameter specific learning rate to tensorboard.
        active_learning : ``Dict[str, int]``, optional, (default = None)
            Settings for active learning, ONLY applies if model is a CorefResolver
        """
        self.model = model
        self.ensemble_model = ensemble_model
        self.ensemble_optimizer = ensemble_optimizer
        self.ensemble_scheduler = ensemble_scheduler
        # start by initializing model to 0th one
        self.model_idx = 0
        if self.ensemble_model is not None:
            self.model = self.ensemble_model.submodels[self.model_idx]
        self.iterator = iterator
        self._held_out_iterator = held_out_iterator
        self._validation_iterator = validation_iterator
        self.shuffle = shuffle
        self.optimizer = optimizer
        self.train_data = train_dataset
        self._held_out_train_data = held_out_train_dataset
        self._discrete_query_time_info = None
        self._discrete_query_time_diff = 0  # our time - standard time
        self._equal_time_flag = active_learning.get('use_equal_annot_time', False)
        if self._equal_time_flag:
            if os.path.exists(active_learning['equal_annot_time_file']):
                with open(active_learning['equal_annot_time_file']) as f:
                    self._discrete_query_time_info = json.load(f)
        self._docid_to_query_time_info = {}
        self._validation_data = validation_dataset

        if patience is None:  # no early stopping
            if validation_dataset:
                logger.warning('You provided a validation dataset but patience was set to None, '
                               'meaning that early stopping is disabled')
        elif (not isinstance(patience, int)) or patience <= 0:
            raise ConfigurationError('{} is an invalid value for "patience": it must be a positive integer '
                                     'or None (if you want to disable early stopping)'.format(patience))
        self._patience = patience
        self._num_epochs = num_epochs

        self._serialization_dir = serialization_dir
        self._num_serialized_models_to_keep = num_serialized_models_to_keep
        self._keep_serialized_model_every_num_seconds = keep_serialized_model_every_num_seconds
        self._serialized_paths: List[Any] = []
        self._last_permanent_saved_checkpoint_time = time.time()
        self._model_save_interval = model_save_interval

        self._grad_norm = grad_norm
        self._grad_clipping = grad_clipping
        self._learning_rate_scheduler = learning_rate_scheduler

        increase_or_decrease = validation_metric[0]
        if increase_or_decrease not in ["+", "-"]:
            raise ConfigurationError("Validation metrics must specify whether they should increase "
                                     "or decrease by pre-pending the metric name with a +/-.")
        self._validation_metric = validation_metric[1:]
        self._validation_metric_decreases = increase_or_decrease == "-"

        if not isinstance(cuda_device, int):
            raise ConfigurationError("Expected an int for cuda_device, got {}".format(cuda_device))

        self._multiple_gpu = False
        self._cuda_devices = [cuda_device]

        if self._cuda_devices[0] != -1:
            self.model = self.model.cuda(self._cuda_devices[0])
            if self.ensemble_model:
                self.ensemble_model = self.ensemble_model.cuda(self._cuda_devices[0])

        self._log_interval = 10  # seconds
        self._summary_interval = summary_interval
        self._histogram_interval = histogram_interval
        self._log_histograms_this_batch = False
        self._should_log_parameter_statistics = should_log_parameter_statistics
        self._should_log_learning_rate = should_log_learning_rate

        # We keep the total batch number as a class variable because it
        # is used inside a closure for the hook which logs activations in
        # ``_enable_activation_logging``.
        self._batch_num_total = 0

        self._last_log = 0.0  # time of last logging

        if serialization_dir is not None:
            train_log = SummaryWriter(os.path.join(serialization_dir, "log", "train"))
            validation_log = SummaryWriter(os.path.join(serialization_dir, "log", "validation"))
            self._tensorboard = TensorboardWriter(train_log, validation_log)
        else:
            self._tensorboard = TensorboardWriter()
        self._warned_tqdm_ignores_underscores = False

        # Whether or not to do active learning
        self._do_active_learning = False
        if active_learning:
            self._do_active_learning = True
            self._active_learning_epoch_interval = active_learning['epoch_interval']
            self._active_learning_num_labels = active_learning['num_labels']
            self._save_al_queries = active_learning['save_al_queries']
            self._active_learning_patience = active_learning['patience']
            self._replace_with_next_pos_edge = active_learning['replace_with_next_pos_edge']
            self._selector = active_learning['selector']['type'] if 'selector' in active_learning else 'entropy'
            if self._selector == 'qbc':
                assert(ensemble_model is not None)
                self._existing_trained = [0]
            self._selector_clusters = active_learning['selector']['use_clusters'] if 'selector' in active_learning else True
            self._query_type = active_learning['query_type'] if 'query_type' in active_learning else 'discrete'
            assert(self._query_type == 'pairwise' or self._query_type == 'discrete')

    def _enable_gradient_clipping(self) -> None:
        if self._grad_clipping is not None:
            # Pylint is unable to tell that we're in the case that _grad_clipping is not None...
            # pylint: disable=invalid-unary-operand-type
            clip_function = lambda grad: grad.clamp(-self._grad_clipping, self._grad_clipping)
            for parameter in self.model.parameters():
                if parameter.requires_grad:
                    parameter.register_hook(clip_function)

    def _enable_activation_logging(self) -> None:
        """
        Log activations to tensorboard
        """
        if self._histogram_interval is not None:
            # To log activation histograms to the forward pass, we register
            # a hook on forward to capture the output tensors.
            # This uses a closure on self._log_histograms_this_batch to
            # determine whether to send the activations to tensorboard,
            # since we don't want them on every call.
            for _, module in self.model.named_modules():
                if not getattr(module, 'should_log_activations', False):
                    # skip it
                    continue

                def hook(module_, inputs, outputs):
                    # pylint: disable=unused-argument,cell-var-from-loop
                    log_prefix = 'activation_histogram/{0}'.format(module_.__class__)
                    if self._log_histograms_this_batch:
                        if isinstance(outputs, torch.Tensor):
                            log_name = log_prefix
                            self._tensorboard.add_train_histogram(log_name,
                                                                  outputs,
                                                                  self._batch_num_total)
                        elif isinstance(outputs, (list, tuple)):
                            for i, output in enumerate(outputs):
                                log_name = "{0}_{1}".format(log_prefix, i)
                                self._tensorboard.add_train_histogram(log_name,
                                                                      output,
                                                                      self._batch_num_total)
                        elif isinstance(outputs, dict):
                            for k, tensor in outputs.items():
                                log_name = "{0}_{1}".format(log_prefix, k)
                                self._tensorboard.add_train_histogram(log_name,
                                                                      tensor,
                                                                      self._batch_num_total)
                        else:
                            # skip it
                            pass

                module.register_forward_hook(hook)

    def rescale_gradients(self) -> Optional[float]:
        """
        Performs gradient rescaling. Is a no-op if gradient rescaling is not enabled.
        """
        if self._grad_norm:
            parameters_to_clip = [p for p in self.model.parameters()
                                  if p.grad is not None]
            return sparse_clip_norm(parameters_to_clip, self._grad_norm)
        return None

    def batch_loss(self, batch: torch.Tensor, for_training: bool) -> torch.Tensor:
        """
        Does a forward pass on the given batch and returns the ``loss`` value in the result.
        If ``for_training`` is `True` also applies regularization penalty.
        """
        batch = util.move_to_device(batch, self._cuda_devices[0])
        output_dict = self.model(**batch)

        try:
            loss = output_dict["loss"]
            if for_training:
                loss += self.model.get_regularization_penalty()
        except KeyError:
            if for_training:
                raise RuntimeError("The model you are trying to optimize does not contain a"
                                   " 'loss' key in the output of model.forward(inputs).")
            loss = None

        return loss

    def _get_metrics(self, total_loss: float, num_batches: int, reset: bool = False) -> Dict[str, float]:
        """
        Gets the metrics but sets ``"loss"`` to
        the total loss divided by the ``num_batches`` so that
        the ``"loss"`` metric is "average loss per batch".
        """
        metrics = self.model.get_metrics(reset=reset)
        metrics["loss"] = float(total_loss / num_batches) if num_batches > 0 else 0.0
        return metrics

    @retry(stop_max_attempt_number=25)
    def _backprop(self, loss):
        torch.cuda.empty_cache()
        loss.backward(retain_graph=True)

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        logger.info(f"Peak CPU memory usage MB: {peak_memory_mb()}")
        for gpu, memory in gpu_memory_mb().items():
            logger.info(f"GPU {gpu} memory usage MB: {memory}")

        train_loss = 0.0
        # Set the model to "train" mode.
        self.model.train()

        # Get tqdm for the training batches
        train_generator = self.iterator(self.train_data,
                                        num_epochs=1,
                                        shuffle=self.shuffle)
        num_training_batches = self.iterator.get_num_batches(self.train_data)
        self._last_log = time.time()
        last_save_time = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        if self._histogram_interval is not None:
            histogram_parameters = set(self.model.get_parameters_for_histogram_tensorboard_logging())

        logger.info("Training")
        train_generator_tqdm = Tqdm.tqdm(train_generator,
                                         total=num_training_batches)
        for batch in train_generator_tqdm:
            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            self._log_histograms_this_batch = self._histogram_interval is not None and (
                    batch_num_total % self._histogram_interval == 0)

            self.optimizer.zero_grad()

            loss = self.batch_loss(batch, for_training=True)
            try:
                torch.cuda.empty_cache()
                loss.backward(retain_graph=True)
            except:
                self._backprop(loss)

            train_loss += loss.item()

            batch_grad_norm = self.rescale_gradients()

            # This does nothing if batch_num_total is None or you are using an
            # LRScheduler which doesn't update per batch.
            if self._learning_rate_scheduler and (self._held_out_train_data is None or
                                                  len(self._held_out_train_data) == 0):
                self._learning_rate_scheduler.step_batch(batch_num_total)

            if self._log_histograms_this_batch:
                # get the magnitude of parameter updates for logging
                # We need a copy of current parameters to compute magnitude of updates,
                # and copy them to CPU so large models won't go OOM on the GPU.
                param_updates = {name: param.detach().cpu().clone()
                                 for name, param in self.model.named_parameters()}
                self.optimizer.step()
                for name, param in self.model.named_parameters():
                    param_updates[name].sub_(param.detach().cpu())
                    update_norm = torch.norm(param_updates[name].view(-1, ))
                    param_norm = torch.norm(param.view(-1, )).cpu()
                    self._tensorboard.add_train_scalar("gradient_update/" + name,
                                                       update_norm / (param_norm + 1e-7),
                                                       batch_num_total)
            else:
                self.optimizer.step()

            # Update the description with the latest metrics
            metrics = self._get_metrics(train_loss, batches_this_epoch)
            description = self._description_from_metrics(metrics)

            train_generator_tqdm.set_description(description, refresh=False)

            # Log parameter values to Tensorboard
            if batch_num_total % self._summary_interval == 0:
                if self._should_log_parameter_statistics:
                    self._parameter_and_gradient_statistics_to_tensorboard(batch_num_total, batch_grad_norm)
                if self._should_log_learning_rate:
                    self._learning_rates_to_tensorboard(batch_num_total)
                self._tensorboard.add_train_scalar("loss/loss_train", metrics["loss"], batch_num_total)
                self._metrics_to_tensorboard(batch_num_total,
                                             {"epoch_metrics/" + k: v for k, v in metrics.items()})

            if self._log_histograms_this_batch:
                self._histograms_to_tensorboard(batch_num_total, histogram_parameters)

            # Save model if needed.
            if self._model_save_interval is not None and (
                    time.time() - last_save_time > self._model_save_interval
            ):
                last_save_time = time.time()
                self._save_checkpoint(
                        '{0}.{1}'.format(epoch, time_to_str(int(last_save_time))), [], is_best=False
                )

        return self._get_metrics(train_loss, batches_this_epoch, reset=True)

    def _should_stop_early(self, metric_history: List[float], patience: int) -> bool:
        """
        uses patience and the validation metric to determine if training should stop early
        """
        if patience and patience < len(metric_history):
            # Pylint can't figure out that in this branch `self._patience` is an int.
            # pylint: disable=invalid-unary-operand-type

            # Is the best score in the past N epochs worse than or equal the best score overall?
            if self._validation_metric_decreases:
                return min(metric_history[-patience:]) >= min(metric_history[:-patience])
            else:
                return max(metric_history[-patience:]) <= max(metric_history[:-patience])

        return False

    def _parameter_and_gradient_statistics_to_tensorboard(self, # pylint: disable=invalid-name
                                                          epoch: int,
                                                          batch_grad_norm: float) -> None:
        """
        Send the mean and std of all parameters and gradients to tensorboard, as well
        as logging the average gradient norm.
        """
        # Log parameter values to Tensorboard
        for name, param in self.model.named_parameters():
            self._tensorboard.add_train_scalar("parameter_mean/" + name,
                                               param.data.mean(),
                                               epoch)
            self._tensorboard.add_train_scalar("parameter_std/" + name, param.data.std(), epoch)
            if param.grad is not None:
                if is_sparse(param.grad):
                    # pylint: disable=protected-access
                    grad_data = param.grad.data._values()
                else:
                    grad_data = param.grad.data

                # skip empty gradients
                if torch.prod(torch.tensor(grad_data.shape)).item() > 0: # pylint: disable=not-callable
                    self._tensorboard.add_train_scalar("gradient_mean/" + name,
                                                       grad_data.mean(),
                                                       epoch)
                    self._tensorboard.add_train_scalar("gradient_std/" + name,
                                                       grad_data.std(),
                                                       epoch)
                else:
                    # no gradient for a parameter with sparse gradients
                    logger.info("No gradient for %s, skipping tensorboard logging.", name)
        # norm of gradients
        if batch_grad_norm is not None:
            self._tensorboard.add_train_scalar("gradient_norm",
                                               batch_grad_norm,
                                               epoch)

    def _learning_rates_to_tensorboard(self, batch_num_total: int):
        """
        Send current parameter specific learning rates to tensorboard
        """
        # optimizer stores lr info keyed by parameter tensor
        # we want to log with parameter name
        names = {param: name for name, param in self.model.named_parameters()}
        for group in self.optimizer.param_groups:
            if 'lr' not in group:
                continue
            rate = group['lr']
            for param in group['params']:
                # check whether params has requires grad or not
                effective_rate = rate * float(param.requires_grad)
                self._tensorboard.add_train_scalar(
                        "learning_rate/" + names[param],
                        effective_rate,
                        batch_num_total
                )

    def _histograms_to_tensorboard(self, epoch: int, histogram_parameters: Set[str]) -> None:
        """
        Send histograms of parameters to tensorboard.
        """
        for name, param in self.model.named_parameters():
            if name in histogram_parameters:
                self._tensorboard.add_train_histogram("parameter_histogram/" + name,
                                                      param,
                                                      epoch)

    def _metrics_to_tensorboard(self,
                                epoch: int,
                                train_metrics: dict,
                                val_metrics: dict = None) -> None:
        """
        Sends all of the train metrics (and validation metrics, if provided) to tensorboard.
        """
        metric_names = set(train_metrics.keys())
        if val_metrics is not None:
            metric_names.update(val_metrics.keys())
        val_metrics = val_metrics or {}

        for name in metric_names:
            train_metric = train_metrics.get(name)
            if train_metric is not None:
                self._tensorboard.add_train_scalar(name, train_metric, epoch)
            val_metric = val_metrics.get(name)
            if val_metric is not None:
                self._tensorboard.add_validation_scalar(name, val_metric, epoch)

    def _metrics_to_console(self,  # pylint: disable=no-self-use
                            train_metrics: dict,
                            val_metrics: dict = None) -> None:
        """
        Logs all of the train metrics (and validation metrics, if provided) to the console.
        """
        val_metrics = val_metrics or {}
        dual_message_template = "%s |  %8.3f  |  %8.3f"
        no_val_message_template = "%s |  %8.3f  |  %8s"
        no_train_message_template = "%s |  %8s  |  %8.3f"
        header_template = "%s |  %-10s"

        metric_names = set(train_metrics.keys())
        if val_metrics:
            metric_names.update(val_metrics.keys())

        name_length = max([len(x) for x in metric_names])

        logger.info(header_template, "Training".rjust(name_length + 13), "Validation")
        for name in metric_names:
            train_metric = train_metrics.get(name)
            val_metric = val_metrics.get(name)

            if val_metric is not None and train_metric is not None:
                logger.info(dual_message_template, name.ljust(name_length), train_metric, val_metric)
            elif val_metric is not None:
                logger.info(no_train_message_template, name.ljust(name_length), "N/A", val_metric)
            elif train_metric is not None:
                logger.info(no_val_message_template, name.ljust(name_length), train_metric, "N/A")

    def _validation_loss(self) -> Tuple[float, int]:
        """
        Computes the validation loss. Returns it and the number of batches.
        """
        logger.info("Validating")

        self.model.eval()

        if self._validation_iterator is not None:
            val_iterator = self._validation_iterator
        else:
            val_iterator = self.iterator

        val_generator = val_iterator(self._validation_data,
                                     num_epochs=1,
                                     shuffle=False)
        num_validation_batches = val_iterator.get_num_batches(self._validation_data)
        val_generator_tqdm = Tqdm.tqdm(val_generator,
                                       total=num_validation_batches)
        batches_this_epoch = 0
        val_loss = 0
        for batch in val_generator_tqdm:

            torch.cuda.empty_cache()
            loss = self.batch_loss(batch, for_training=False)
            if loss is not None:
                # You shouldn't necessarily have to compute a loss for validation, so we allow for
                # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                # currently only used as the divisor for the loss function, so we can safely only
                # count those batches for which we actually have a loss.  If this variable ever
                # gets used for something else, we might need to change things around a bit.
                batches_this_epoch += 1
                val_loss += loss.detach().cpu().numpy()

            # Update the description with the latest metrics
            val_metrics = self._get_metrics(val_loss, batches_this_epoch)
            description = self._description_from_metrics(val_metrics)
            val_generator_tqdm.set_description(description, refresh=False)

        return val_loss, batches_this_epoch

    def train(self) -> Dict[str, Any]:
        """
        Trains the supplied model with the supplied parameters.
        """
        try:
            epoch_counter, validation_metric_per_epoch = self._restore_checkpoint()
        except RuntimeError:
            traceback.print_exc()
            raise ConfigurationError("Could not recover training from the checkpoint.  Did you mean to output to "
                                     "a different serialization directory or delete the existing serialization "
                                     "directory?")

        self._enable_gradient_clipping()
        self._enable_activation_logging()

        if self._do_active_learning:
            # save initial model state to retrain from scratch
            if self._selector == 'qbc':
                init_model_state = self.ensemble_model.state_dict()
                init_optimizer_state = [optimizer.state_dict() for optimizer in self.ensemble_optimizer]
                init_lr_scheduler_state = [scheduler.lr_scheduler.state_dict() for scheduler in self.ensemble_scheduler]
            else:
                init_model_state = self.model.state_dict()
                init_optimizer_state = self.optimizer.state_dict()
                init_lr_scheduler_state = self._learning_rate_scheduler.lr_scheduler.state_dict()
            init_model_path = os.path.join(self._serialization_dir, "init_model_state.th")
            init_optimizer_path = os.path.join(self._serialization_dir, "init_optimizer_state.th")
            init_lr_scheduler_path = os.path.join(self._serialization_dir, "init_lr_scheduler_state.th")
            torch.save(init_model_state, init_model_path)
            torch.save(init_optimizer_state, init_optimizer_path)
            torch.save(init_lr_scheduler_state, init_lr_scheduler_path)

        self._finished_training_eval_ensemble = False
        if self._selector == 'qbc' and self._finished_training_eval_ensemble:
            metrics = {}
            epoch = 0
            with torch.no_grad():
                # evaluate ensemble of BEST models at this epoch
                for i in range(len(self.ensemble_model.submodels)):
                    submodel_path = os.path.join(self._serialization_dir, "best_submodel_" + str(i) + "_state.th")
                    submodel_state = torch.load(submodel_path, map_location=util.device_mapping(-1))
                    self.ensemble_model.submodels[i].load_state_dict(submodel_state)
                self.model = self.ensemble_model
                # We have a validation set, so compute all the metrics on it.
                val_loss, num_batches = self._validation_loss()
                ensemble_val_metrics = self._get_metrics(val_loss, num_batches, reset=True)

            for key, value in ensemble_val_metrics.items():
                metrics["ensemble_validation_" + key] = value

            if self._serialization_dir:
                dump_metrics(os.path.join(self._serialization_dir, f'metrics_epoch_{epoch}.json'), metrics)
            return metrics, None

        logger.info("Beginning training.")

        train_metrics: Dict[str, float] = {}  # for submodels only
        val_metrics: Dict[str, float] = {}
        ensemble_val_metrics: Dict[str, float] = {}
        metrics: Dict[str, Any] = {}
        # outer list of models, inner list of epochs
        submodel_val_metrics: List[List[float]] = [[] for i in range(len(self.ensemble_model.submodels))] if self.ensemble_model is not None else None
        epochs_trained = 0
        training_start_time = time.time()
        first_epoch_for_converge = 0
        
        max_epoch = self._num_epochs
        if self.ensemble_model is not None:
            max_epoch *= len(self.ensemble_model.submodels)
        for epoch in range(epoch_counter, max_epoch):
            epoch_start_time = time.time()
            if self.ensemble_model is not None:
                self.model = self.ensemble_model.submodels[self.model_idx]
                self.optimizer = self.ensemble_optimizer[self.model_idx]
                self._learning_rate_scheduler = self.ensemble_scheduler[self.model_idx]

            train_metrics = self._train_epoch(epoch)
            query_this_epoch = False
            increment_model = False
            all_finished = False

            if self._validation_data is not None:
                with torch.no_grad():
                    eval_ensemble = False
                    # We have a validation set, so compute all the metrics on it.
                    val_loss, num_batches = self._validation_loss()
                    val_metrics = self._get_metrics(val_loss, num_batches, reset=True)

                    # Check validation metric for early stopping
                    this_epoch_val_metric = val_metrics[self._validation_metric]

                    # Check validation metric to see if it's the best so far
                    if self._selector == 'qbc':
                        is_best_so_far = self._is_best_so_far(this_epoch_val_metric, submodel_val_metrics[self.model_idx])
                        submodel_val_metrics[self.model_idx].append(this_epoch_val_metric)
                    else:
                        is_best_so_far = self._is_best_so_far(this_epoch_val_metric, validation_metric_per_epoch)
                    validation_metric_per_epoch.append(this_epoch_val_metric)

                    # check convergence (determine whether or not to query this epoch)
                    if self._do_active_learning and len(self._held_out_train_data) > 0:
                        if self._should_stop_early(
                            validation_metric_per_epoch[first_epoch_for_converge:], self._active_learning_patience
                        ) or (epoch - first_epoch_for_converge >= self._active_learning_epoch_interval):
                            logger.info("Ran out of patience on model " + str(self.model_idx))
                            if self._selector == 'qbc':
                                first_epoch_for_converge = epoch + 1
                                increment_model = True
                            if self._selector != 'qbc' or self.model_idx == len(self.ensemble_model.submodels) - 1: # at last model, or only 1 model
                                # still have more data to add
                                query_this_epoch = True
                                eval_ensemble = (self._selector == 'qbc')
                                logger.info("Evaluating ensemble and adding more data.")
                    else:
                        if self._should_stop_early(validation_metric_per_epoch[first_epoch_for_converge:], self._patience) or (
                            submodel_val_metrics is not None and len(submodel_val_metrics[self.model_idx]) >= self._num_epochs
                        ):
                            logger.info("Ran out of patience on model " + str(self.model_idx))
                            if self._selector == 'qbc':
                                first_epoch_for_converge = epoch + 1
                                increment_model = True
                            if self._selector != 'qbc' or self.model_idx == len(self.ensemble_model.submodels) - 1: # at last model, or only 1 model
                                eval_ensemble = (self._selector == 'qbc')
                                logger.info("Evaluating ensemble and stopping training.")
                                all_finished = True
                    # TODO
                    # query_this_epoch |= (loaded_init_converged and epoch == 0)  # loaded the initial converged model
            else:
                # No validation set, so just assume it's the best so far.
                is_best_so_far = True
                val_metrics = {}
                this_epoch_val_metric = None

            self._metrics_to_tensorboard(epoch, train_metrics, val_metrics=val_metrics)
            self._metrics_to_console(train_metrics, val_metrics)

            # Create overall metrics dict
            training_elapsed_time = time.time() - training_start_time
            metrics["training_duration"] = time.strftime("%H:%M:%S", time.gmtime(training_elapsed_time))
            metrics["training_start_epoch"] = epoch_counter
            metrics["training_epochs"] = epochs_trained
            metrics["epoch"] = epoch

            if is_best_so_far:
                # Update all the best_ metrics.
                # (Otherwise they just stay the same as they were.)
                metrics['best_epoch'] = epoch
                for key, value in val_metrics.items():
                    metrics["best_validation_" + key] = value
                # save the best model (already incremented self.model_idx, so -1 here)
                submodel_path = os.path.join(self._serialization_dir, "best_submodel_" + str(self.model_idx) + "_state.th")
                torch.save(self.model.state_dict(), submodel_path)

            if self._validation_data is not None and eval_ensemble:
                with torch.no_grad():
                    # evaluate ensemble of BEST models at this epoch
                    for i in range(len(self.ensemble_model.submodels)):
                        submodel_path = os.path.join(self._serialization_dir, "best_submodel_" + str(i) + "_state.th")
                        submodel_state = torch.load(submodel_path, map_location=util.device_mapping(-1))
                        self.ensemble_model.submodels[i].load_state_dict(submodel_state)
                    self.model = self.ensemble_model
                    # We have a validation set, so compute all the metrics on it.
                    val_loss, num_batches = self._validation_loss()
                    ensemble_val_metrics = self._get_metrics(val_loss, num_batches, reset=True)

            for key, value in train_metrics.items():
                metrics["training_" + key] = value
            for key, value in val_metrics.items():
                metrics["validation_" + key] = value
            for key, value in ensemble_val_metrics.items():
                metrics["ensemble_validation_" + key] = value

            if self._serialization_dir:
                dump_metrics(os.path.join(self._serialization_dir, f'metrics_epoch_{epoch}.json'), metrics)
            if self._learning_rate_scheduler and (self._held_out_train_data is None or
                                                  len(self._held_out_train_data) == 0):
                # The LRScheduler API is agnostic to whether your schedule requires a validation metric -
                # if it doesn't, the validation metric passed here is ignored.
                self._learning_rate_scheduler.step(this_epoch_val_metric, epoch)

            self._save_checkpoint(epoch, validation_metric_per_epoch, is_best=is_best_so_far)

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", time.strftime("%H:%M:%S", time.gmtime(epoch_elapsed_time)))

            if epoch < self._num_epochs - 1:
                training_elapsed_time = time.time() - training_start_time
                estimated_time_remaining = training_elapsed_time * \
                    ((self._num_epochs - epoch_counter) / float(epoch - epoch_counter + 1) - 1)
                formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
                logger.info("Estimated training time remaining: %s", formatted_time)

            if increment_model:
                self.model_idx = (self.model_idx + 1) % len(self.ensemble_model.submodels)
            if all_finished:
                break

            # ''' ACTIVE LEARNING BY SELF-TRAINING/EM:
            # 1. evaluate on held-out training data
            # 2. use active learning/gold labels to confirm/deny labels on held-out training data
            # 3. add correct instances in held-out training data to actual train data, then re-train
            if self._do_active_learning and len(self._held_out_train_data) > 0 and query_this_epoch:
                # take a subset of training data to evaluate on, and add to actual training set
                train_data_to_add = self._held_out_train_data[:280]
                self._held_out_train_data = self._held_out_train_data[280:]
                held_out_generator = self._held_out_iterator(train_data_to_add, num_epochs=1, shuffle=False)
                num_held_out_batches = self.iterator.get_num_batches(train_data_to_add)
                held_out_generator_tqdm = Tqdm.tqdm(held_out_generator, total=num_held_out_batches)
                conll_coref = ConllCorefScores()
                total_labels = 0
                total_num_queried = 0

                with torch.no_grad():
                    logger.info("Held-Out Training")
                    # Run model on held out training data
                    self.model.eval()

                    num_batches = 0
                    held_out_loss = 0
                    for batch_ind, batch in enumerate(held_out_generator_tqdm):
                        # transfer all of batch to GPU
                        for key in batch:
                            if isinstance(batch[key], torch.Tensor):
                                batch[key] = batch[key].cuda(self._cuda_devices[0])
                        batch['get_scores'] = True
                        if self._selector == 'qbc':
                            self.model = self.ensemble_model
                        batch = util.move_to_device(batch, self._cuda_devices[0])
                        output_dict = self.model(**batch)
                        batch_size = len(output_dict['predicted_antecedents'])
                        translation_reference = output_dict['top_span_indices']

                        # create must-links and cannot-links
                        if ('must_link' not in batch) or batch['must_link'] is None:
                            batch['must_link'] = torch.empty(0, dtype=torch.long).cuda(self._cuda_devices[0])
                        if ('cannot_link' not in batch) or batch['cannot_link'] is None:
                            batch['cannot_link'] = torch.empty(0, dtype=torch.long).cuda(self._cuda_devices[0])

                        # history of mentions that have already been queried/exist in gold data (index in top_spans)
                        if self._query_type == 'discrete':
                            all_queried_mentions = (batch['span_labels'] != -1).nonzero()
                            queried_mentions_mask = torch.zeros(output_dict['coreference_scores'].size()[:2],
                                dtype=torch.bool).cuda(self._cuda_devices[0])  # should be all false
                            # convert to indices of top_spans for consistency's sake
                            if len(all_queried_mentions) > 0:
                                all_queried_mentions_spans = batch['spans'][all_queried_mentions[:,0], all_queried_mentions[:,1]]
                                top_queried_mentions_spans = ((all_queried_mentions_spans.unsqueeze(1) - output_dict['top_spans']).abs().sum(-1) == 0).nonzero()
                                batch_inds = all_queried_mentions[top_queried_mentions_spans[:,0]][:,0]
                                # ASSUMES 1 INSTANCE/BATCH
                                queried_mentions_mask[batch_inds, top_queried_mentions_spans[:,1]] = 1
                        else:  # query type is pairwise
                            all_queried_edges = (batch['span_labels'] != -1).nonzero()
                            queried_edges_mask = torch.zeros(output_dict['coreference_scores'].size(),
                                                                dtype=torch.bool).cuda(self._cuda_devices[0])
                            if len(all_queried_edges) > 0:
                                top_queried_edges = al_util.translate_to_indA(all_queried_edges, output_dict,
                                                                                batch['spans'],
                                                                                translation_reference=translation_reference)
                                queried_edges_mask[top_queried_edges[:, 0], top_queried_edges[:, 1],
                                                    top_queried_edges[:, 2]] = 1
                            queried_edges_mask[:, :, 0] = 1  # don't query 1st column (empty)
                            queried_edges_mask |= output_dict['coreference_scores'] == -float('inf')

                        confirmed_clusters = batch['span_labels'].clone()
                        confirmed_non_coref_edges = torch.tensor([], dtype=torch.long).cuda(self._cuda_devices[0])

                        # Update span_labels with model-predicted clusters
                        output_dict = self.model.decode(output_dict)
                        has_antecedent_mask = (output_dict['predicted_antecedents'] != -1)
                        model_edges = torch.empty(0, dtype=torch.long).cuda(self._cuda_devices[0])
                        if len(has_antecedent_mask.nonzero()) > 0:
                            model_edges = torch.cat((has_antecedent_mask.nonzero(), output_dict['predicted_antecedents'][has_antecedent_mask].unsqueeze(-1)), dim=-1)
                        indA_model_edges = al_util.translate_to_indA(model_edges, output_dict, batch['spans'], translation_reference=translation_reference)
                        for edge in indA_model_edges:
                            batch['span_labels'] = al_util.update_clusters_with_edge(batch['span_labels'], edge)

                        if self._query_type == 'discrete':
                            total_possible_queries = len(output_dict['top_spans'][0])
                            if self._discrete_query_time_info is not None:
                                # ONLY FOR 1 INSTANCE PER BATCH
                                batch_query_info = self._discrete_query_time_info[batch['metadata'][0]["ID"]]
                                num_not_coref = batch_query_info['num_queried'] - batch_query_info.get('coref', 0)
                                self._discrete_query_time_diff -= (
                                    num_not_coref * DISCRETE_Q_TIME_TOTAL + batch_query_info.get('coref', 0) * PAIRWISE_Q_TIME
                                )
                                assert batch_query_info['batch_size'] == 1
                                # min(total_possible, # that can be queried if all answered positively)
                                num_to_query = min(total_possible_queries, int(math.ceil(
                                    num_not_coref * DISCRETE_PAIRWISE_RATIO + batch_query_info.get('coref', 0)
                                )))
                            else:
                                num_to_query = min(self._active_learning_num_labels, total_possible_queries)
                            top_spans_model_labels = torch.gather(batch['span_labels'], 1, translation_reference)

                            # score selector stuff
                            verify_existing = None
                            if self._selector == 'score':
                                verify_existing = True

                            num_queried = 0
                            num_coreferent = 0
                            while num_queried < num_to_query:
                                if self._discrete_query_time_info is not None and self._discrete_query_time_diff >= 0:
                                    break
                                # num existing edges to verify = min((# to query total / 3), # of existing edges)
                                if self._selector == 'score' and (num_queried == int(num_to_query / 2) or num_queried == len(model_edges)):
                                    verify_existing = False
                                if self._selector_clusters:
                                    mention, mention_score = \
                                        al_util.find_next_most_uncertain_mention(self._selector,
                                                                                top_spans_model_labels,
                                                                                output_dict, queried_mentions_mask,
                                                                                verify_existing=verify_existing)
                                else:
                                    mention, mention_score = \
                                        al_util.find_next_most_uncertain_mention_unclustered(self._selector,
                                                                                    top_spans_model_labels,
                                                                                    output_dict, queried_mentions_mask,
                                                                                    verify_existing=verify_existing)
                                indA_edge, edge_asked, indA_edge_asked = \
                                    al_util.query_user_labels_mention(mention, output_dict, batch['spans'],
                                                                        batch['user_labels'], translation_reference,
                                                                        self._save_al_queries, batch,
                                                                        os.path.join(self._serialization_dir, "saved_queries_epoch_{}.json".format(epoch)),
                                                                        batch['span_labels'])
                                if indA_edge_asked[2] == indA_edge[2]:
                                    self._discrete_query_time_diff += PAIRWISE_Q_TIME
                                    num_coreferent += 1
                                else:
                                    self._discrete_query_time_diff += DISCRETE_Q_TIME_TOTAL

                                # add mention to queried before (arbitrarily set it in predicted_antecedents and coreference_scores to no cluster, even if not truly
                                # the case--the only thing that matters is that it has a value that it is 100% confident of)
                                queried_mentions_mask[mention[0], mention[1]] = 1

                                # If asked edge was deemed not coreferent, delete it
                                if indA_edge_asked[2] != indA_edge[2]:
                                    if len(indA_model_edges) > 0:
                                        # (both lines below implicitly check whether indA_edge_asked was actually added before)
                                        edge_asked_mask = (indA_model_edges == indA_edge_asked).sum(-1)
                                        batch['span_labels'] = al_util.update_clusters_with_edge(
                                            batch['span_labels'], indA_edge_asked, delete=True,
                                            all_edges=indA_model_edges)
                                        indA_model_edges = indA_model_edges[edge_asked_mask < 3]
                                    # Add to confirmed non-coreferent
                                    if len(confirmed_non_coref_edges) == 0:
                                        confirmed_non_coref_edges = indA_edge_asked.unsqueeze(0)
                                    else:
                                        confirmed_non_coref_edges = torch.cat(
                                            (confirmed_non_coref_edges, indA_edge_asked.unsqueeze(0)), dim=0)
                                    batch['must_link'], batch['cannot_link'], confirmed_clusters, output_dict = \
                                        al_util.get_link_closures_edge(batch['must_link'], batch['cannot_link'],
                                                                        indA_edge_asked, False,
                                                                        confirmed_clusters, output_dict,
                                                                        translation_reference)

                                # Add edge deemed coreferent
                                if indA_edge[2] != -1:
                                    # Add new edge deemed coreferent, if not already in there
                                    if len(indA_model_edges) == 0 or (
                                            (indA_model_edges == indA_edge).sum(1) == 3).sum() == 0:
                                        indA_model_edges = torch.cat((indA_model_edges, indA_edge.unsqueeze(0)),
                                                                        dim=0)
                                        batch['span_labels'] = al_util.update_clusters_with_edge(
                                            batch['span_labels'], indA_edge)
                                    batch['must_link'], batch['cannot_link'], confirmed_clusters, output_dict = \
                                        al_util.get_link_closures_edge(batch['must_link'], batch['cannot_link'],
                                                                        indA_edge, True, confirmed_clusters,
                                                                        output_dict, translation_reference)
                                else:
                                    # set to null antecedent
                                    output_dict['predicted_antecedents'][mention[0], mention[1]] = -1
                                    output_dict['coreference_scores'][mention[0], mention[1], 1:] = -float("inf")
                                    if self._selector == 'qbc':
                                        # must update for each model
                                        output_dict['coreference_scores_models'][:, mention[0], mention[1],
                                        1:] = -float("inf")
                                num_queried += 1
                            for i in range(batch_size):
                                self._docid_to_query_time_info[batch['metadata'][i]["ID"]] = \
                                    {"num_queried": num_queried, "coref": num_coreferent, "not_coref":
                                        num_queried - num_coreferent, "batch_size": batch_size}
                        else:  # pairwise
                            total_possible_queries = len((~queried_edges_mask).nonzero())
                            if self._discrete_query_time_info is not None:
                                # ONLY FOR 1 INSTANCE PER BATCH
                                batch_query_info = self._discrete_query_time_info[batch['metadata'][0]["ID"]]
                                #BOOKMARK
                                num_not_coref = batch_query_info['num_queried'] - batch_query_info.get('coref', 0)
                                self._discrete_query_time_diff -= (
                                    num_not_coref * DISCRETE_Q_TIME_TOTAL + batch_query_info.get('coref', 0) * PAIRWISE_Q_TIME
                                )
                                assert batch_query_info['batch_size'] == 1
                                num_to_query = int(np.round(
                                    num_not_coref * DISCRETE_PAIRWISE_RATIO + batch_query_info.get('coref', 0)
                                ))
                            else:
                                num_to_query = min(self._active_learning_num_labels, total_possible_queries)
                            top_spans_model_labels = torch.gather(batch['span_labels'], 1, translation_reference)
                            num_queried = 0
                            while num_queried < num_to_query:
                                if self._discrete_query_time_info is not None and self._discrete_query_time_diff >= 0:
                                    break
                                edge, edge_score = \
                                    al_util.find_next_most_uncertain_pairwise_edge(self._selector,
                                                                                    top_spans_model_labels,
                                                                                    output_dict, queried_edges_mask)
                                coreferent, indA_edge = \
                                    al_util.query_user_labels_pairwise(edge, output_dict, batch['spans'],
                                                                        batch['user_labels'], translation_reference,
                                                                        self._save_al_queries, batch,
                                                                        os.path.join(self._serialization_dir, "saved_queries_epoch_{}.json".format(epoch)),
                                                                        batch['span_labels'])
                                queried_edges_mask[edge[0], edge[1], edge[2] + 1] = 1
                                # arbitrarily set to null antecedent
                                output_dict['predicted_antecedents'][edge[0], edge[1]] = edge[2]
                                output_dict['coreference_scores'][edge[0], edge[1], edge[2] + 1] = -float("inf")
                                if self._selector == 'qbc':
                                    # must update for each model
                                    output_dict['coreference_scores_models'][:, edge[0], edge[1], edge[2] + 1] = \
                                        -float("inf")

                                if not coreferent:
                                    if len(indA_model_edges) > 0:
                                        # If asked edge was deemed not coreferent, delete it
                                        # (both lines below implicitly check whether indA_edge_asked was actually added before)
                                        edge_asked_mask = (indA_model_edges == indA_edge).sum(1)
                                        batch['span_labels'] = al_util.update_clusters_with_edge(
                                            batch['span_labels'], indA_edge, delete=True,
                                            all_edges=indA_model_edges)
                                        indA_model_edges = indA_model_edges[edge_asked_mask < 3]
                                    # Add to confirmed non-coreferent
                                    if len(confirmed_non_coref_edges) == 0:
                                        confirmed_non_coref_edges = indA_edge.unsqueeze(0)
                                    else:
                                        confirmed_non_coref_edges = torch.cat(
                                            (confirmed_non_coref_edges, indA_edge.unsqueeze(0)), dim=0)
                                    # Add to cannot-link
                                    batch['cannot_link'] = torch.cat((batch['cannot_link'], indA_edge.unsqueeze(0)), dim=0)
                                else:
                                    # Otherwise, add edge, if not already in there
                                    if len(indA_model_edges) == 0 or (
                                            (indA_model_edges == indA_edge).sum(1) == 3).sum() == 0:
                                        indA_model_edges = torch.cat((indA_model_edges, indA_edge.unsqueeze(0)),
                                                                        dim=0)
                                        batch['span_labels'] = al_util.update_clusters_with_edge(
                                            batch['span_labels'], indA_edge)
                                    confirmed_clusters = al_util.update_clusters_with_edge(confirmed_clusters,
                                                                                            indA_edge)
                                    # Add to must-link
                                    batch['must_link'] = torch.cat((batch['must_link'], indA_edge.unsqueeze(0)), dim=0)
                                num_queried += 1

                            for i in range(batch_size):
                                self._docid_to_query_time_info[batch['metadata'][i]["ID"]] = \
                                    {"num_queried": num_queried,  "batch_size": batch_size}

                        edges_to_add = indA_model_edges

                        # keep track of which instances we have to update in training data
                        train_instances_to_update = {}
                        # Update gold clusters based on (corrected) model edges, in span_labels
                        for edge in edges_to_add:
                            if self._query_type != 'discrete':
                                batch['span_labels'] = al_util.update_clusters_with_edge(batch['span_labels'], edge)
                            ind_instance = edge[0].item()  # index of instance in batch
                            if ind_instance not in train_instances_to_update:
                                # [[mustlinks], [cannotlinks]]
                                train_instances_to_update[ind_instance] = [[], []]

                        # do pairwise transitive closure of must-links and cannot-links
                        if self._query_type != 'discrete':
                            batch['must_link'], batch['cannot_link'] = al_util.get_link_closures(batch['must_link'],
                                                                                                    batch['cannot_link'])

                        # update must-links and cannot-links
                        for edge in batch['must_link']:
                            ind_instance = edge[0].item()
                            ind_instance_overall = batch_ind * batch_size + ind_instance  # index in entire train data
                            if ind_instance not in train_instances_to_update:
                                # [[mustlinks], [cannotlinks]]
                                train_instances_to_update[ind_instance] = [[], []]
                            train_instances_to_update[ind_instance][0].append(
                                PairField(
                                    IndexField(edge[1].item(), train_data_to_add[ind_instance_overall].fields['spans']),
                                    IndexField(edge[2].item(), train_data_to_add[ind_instance_overall].fields['spans']),
                                )
                            )
                        for edge in batch['cannot_link']:
                            ind_instance = edge[0].item()
                            ind_instance_overall = batch_ind * batch_size + ind_instance  # index in entire train data
                            if ind_instance not in train_instances_to_update:
                                # [[mustlinks], [cannotlinks]]
                                train_instances_to_update[ind_instance] = [[], []]
                            train_instances_to_update[ind_instance][1].append(
                                PairField(
                                    IndexField(edge[1].item(), train_data_to_add[ind_instance_overall].fields['spans']),
                                    IndexField(edge[2].item(), train_data_to_add[ind_instance_overall].fields['spans']),
                                )
                            )

                        # update train data itself
                        for ind_instance in train_instances_to_update:
                            ind_instance_overall = batch_ind * batch_size + ind_instance  # index in entire train data
                            train_data_to_add[ind_instance_overall].fields['span_labels'] = SequenceLabelField(
                                batch['span_labels'][ind_instance].tolist(),
                                train_data_to_add[ind_instance_overall].fields['span_labels'].sequence_field
                            )
                            if len(train_instances_to_update[ind_instance][0]) == 0:
                                train_data_to_add[ind_instance_overall].fields['must_link'] = ListField([
                                    PairField(IndexField(-1, SequenceField()), IndexField(-1, SequenceField()))
                                ])
                            else:
                                train_data_to_add[ind_instance_overall].fields['must_link'] = ListField(
                                    train_instances_to_update[ind_instance][0]
                                )
                            if len(train_instances_to_update[ind_instance][1]) == 0:
                                train_data_to_add[ind_instance_overall].fields['cannot_link'] = ListField([
                                    PairField(IndexField(-1, SequenceField()), IndexField(-1, SequenceField()))
                                ])
                            else:
                                train_data_to_add[ind_instance_overall].fields['cannot_link'] = ListField(
                                    train_instances_to_update[ind_instance][1]
                                )

                        if output_dict['loss'] is not None:
                            num_batches += 1
                            held_out_loss += output_dict['loss'].detach().cpu().numpy()

                        # COMPUTE METRICS
                        # Update the description with the latest metrics
                        # reset metrics at last epoch
                        held_out_metrics = self._get_metrics(held_out_loss, num_batches, reset=batch_ind == len(held_out_generator_tqdm) - 1)
                        for i, metadata in enumerate(batch['metadata']):
                            predicted_clusters = []
                            for cluster in range(batch['span_labels'][i].max() + 1):
                                # convert spans to tuples
                                predicted_clusters.append(batch['spans'][i][batch['span_labels'][i] == cluster].tolist())
                            predicted_clusters, mention_to_predicted = conll_coref.get_gold_clusters(predicted_clusters)
                            gold_clusters, mention_to_gold = conll_coref.get_gold_clusters(batch['metadata'][i]['clusters'])
                            for scorer in conll_coref.scorers:
                                scorer.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
                        new_P, new_R, new_F1 = conll_coref.get_metric()
                        description_display = {'old_P': held_out_metrics['coref_precision'], 'new_P': new_P,
                                                'old_R': held_out_metrics['coref_recall'], 'new_R': new_R,
                                                'old_F1': held_out_metrics['coref_f1'], 'new_F1': new_F1,
                                                'MR': held_out_metrics['mention_recall'], 'loss': held_out_metrics['loss']}
                        description = self._description_from_metrics(description_display)
                        total_num_queried += num_to_query
                        total_labels += total_possible_queries
                        description += ' # labels: ' + str(total_num_queried) + '/' + str(total_labels) + ' ||'
                        held_out_generator_tqdm.set_description(description, refresh=False)

                # add instance(s) from held-out training dataset to actual dataset (already removed from held-out
                # above)
                self.train_data.extend(train_data_to_add)

                first_epoch_for_converge = epoch + 1

                # at last epoch, retrain from scratch, resetting model params to intial state
                if len(self._held_out_train_data) == 0:
                    init_model_state = torch.load(init_model_path, map_location=util.device_mapping(-1))
                    init_optimizer_state = torch.load(init_optimizer_path, map_location=util.device_mapping(-1))
                    init_scheduler_state = torch.load(init_lr_scheduler_path, map_location=util.device_mapping(-1))
                    if self.ensemble_model is not None:
                        self.ensemble_model.load_state_dict(init_model_state)
                        for i in range(len(self.ensemble_optimizer)):
                            self.ensemble_optimizer[i].load_state_dict(init_optimizer_state[i])
                            self.ensemble_scheduler[i].lr_scheduler.load_state_dict(init_scheduler_state[i])
                            move_optimizer_to_cuda(self.ensemble_optimizer[i])
                        self.optimizer = self.ensemble_optimizer[0]
                        self._learning_rate_scheduler = self.ensemble_scheduler[0]
                        self.model = self.ensemble_model.submodels[0]
                        self.model_idx = 0
                    else:
                        self.model.load_state_dict(init_model_state)
                        self.optimizer.load_state_dict(init_optimizer_state)
                        move_optimizer_to_cuda(self.optimizer)
                        self._learning_rate_scheduler.lr_scheduler.load_state_dict(init_scheduler_state)
            epochs_trained += 1

        return metrics, self._docid_to_query_time_info

    def _is_best_so_far(self,
                        this_epoch_val_metric: float,
                        validation_metric_per_epoch: List[float]):
        if not validation_metric_per_epoch:
            return True
        elif self._validation_metric_decreases:
            return this_epoch_val_metric < min(validation_metric_per_epoch)
        else:
            return this_epoch_val_metric > max(validation_metric_per_epoch)

    def _description_from_metrics(self, metrics: Dict[str, float]) -> str:
        if (not self._warned_tqdm_ignores_underscores and
                    any(metric_name.startswith("_") for metric_name in metrics)):
            logger.warning("Metrics with names beginning with \"_\" will "
                           "not be logged to the tqdm progress bar.")
            self._warned_tqdm_ignores_underscores = True
        return ', '.join(["%s: %.4f" % (name, value) for name, value in
                          metrics.items() if not name.startswith("_")]) + " ||"

    def _save_checkpoint(self,
                         epoch: Union[int, str],
                         val_metric_per_epoch: List[float],
                         is_best: Optional[bool] = None) -> None:
        """
        Saves a checkpoint of the model to self._serialization_dir.
        Is a no-op if self._serialization_dir is None.

        Parameters
        ----------
        epoch : Union[int, str], required.
            The epoch of training.  If the checkpoint is saved in the middle
            of an epoch, the parameter is a string with the epoch and timestamp.
        is_best: bool, optional (default = None)
            A flag which causes the model weights at the given epoch to
            be copied to a "best.th" file. The value of this flag should
            be based on some validation metric computed by your model.
        """
        if self._serialization_dir is not None:
            model_path = os.path.join(self._serialization_dir, "model_state_epoch_{}.th".format(epoch))
            model_state = self.model.state_dict()
            torch.save(model_state, model_path)

            training_state = {'epoch': epoch,
                              'val_metric_per_epoch': val_metric_per_epoch,
                              'optimizer': self.optimizer.state_dict(),
                              'batch_num_total': self._batch_num_total}
            if self._learning_rate_scheduler is not None:
                training_state["learning_rate_scheduler"] = \
                    self._learning_rate_scheduler.lr_scheduler.state_dict()
            training_path = os.path.join(self._serialization_dir,
                                         "training_state_epoch_{}.th".format(epoch))
            torch.save(training_state, training_path)
            if is_best:
                logger.info("Best validation performance so far. "
                            "Copying weights to '%s/best.th'.", self._serialization_dir)
                shutil.copyfile(model_path, os.path.join(self._serialization_dir, "best.th"))

            if self._num_serialized_models_to_keep and self._num_serialized_models_to_keep >= 0:
                self._serialized_paths.append([time.time(), model_path, training_path])
                if len(self._serialized_paths) > self._num_serialized_models_to_keep:
                    paths_to_remove = self._serialized_paths.pop(0)
                    # Check to see if we should keep this checkpoint, if it has been longer
                    # then self._keep_serialized_model_every_num_seconds since the last
                    # kept checkpoint.
                    remove_path = True
                    if self._keep_serialized_model_every_num_seconds is not None:
                        save_time = paths_to_remove[0]
                        time_since_checkpoint_kept = save_time - self._last_permanent_saved_checkpoint_time
                        if time_since_checkpoint_kept > self._keep_serialized_model_every_num_seconds:
                            # We want to keep this checkpoint.
                            remove_path = False
                            self._last_permanent_saved_checkpoint_time = save_time
                    if remove_path:
                        for fname in paths_to_remove[1:]:
                            os.remove(fname)

    def find_latest_checkpoint(self) -> Tuple[str, str]:
        """
        Return the location of the latest model and training state files.
        If there isn't a valid checkpoint then return None.
        """
        have_checkpoint = (self._serialization_dir is not None and
                           any("model_state_epoch_" in x for x in os.listdir(self._serialization_dir)))

        if not have_checkpoint:
            return None

        serialization_files = os.listdir(self._serialization_dir)
        model_checkpoints = [x for x in serialization_files if "model_state_epoch" in x]
        # Get the last checkpoint file.  Epochs are specified as either an
        # int (for end of epoch files) or with epoch and timestamp for
        # within epoch checkpoints, e.g. 5.2018-02-02-15-33-42
        found_epochs = [
                # pylint: disable=anomalous-backslash-in-string
                re.search("model_state_epoch_([0-9\.\-]+)\.th", x).group(1)
                for x in model_checkpoints
        ]
        int_epochs: Any = []
        for epoch in found_epochs:
            pieces = epoch.split('.')
            if len(pieces) == 1:
                # Just a single epoch without timestamp
                int_epochs.append([int(pieces[0]), 0])
            else:
                # has a timestamp
                int_epochs.append([int(pieces[0]), pieces[1]])
        last_epoch = sorted(int_epochs, reverse=True)[0]
        if last_epoch[1] == 0:
            epoch_to_load = str(last_epoch[0])
        else:
            epoch_to_load = '{0}.{1}'.format(last_epoch[0], last_epoch[1])

        model_path = os.path.join(self._serialization_dir,
                                  "model_state_epoch_{}.th".format(epoch_to_load))
        training_state_path = os.path.join(self._serialization_dir,
                                           "training_state_epoch_{}.th".format(epoch_to_load))

        return (model_path, training_state_path)

    def _restore_checkpoint(self) -> Tuple[int, List[float]]:
        """
        Restores a model from a serialization_dir to the last saved checkpoint.
        This includes an epoch count and optimizer state, which is serialized separately
        from  model parameters. This function should only be used to continue training -
        if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        `` model.load_state_dict(torch.load("/path/to/model/weights.th"))``

        If ``self._serialization_dir`` does not exist or does not contain any checkpointed weights,
        this function will do nothing and return 0.

        Returns
        -------
        epoch: int
            The epoch at which to resume training, which should be one after the epoch
            in the saved training state.
        """
        latest_checkpoint = self.find_latest_checkpoint()

        if latest_checkpoint is None:
            # No checkpoint to restore, start at 0
            return 0, []

        model_path, training_state_path = latest_checkpoint

        # Load the parameters onto CPU, then transfer to GPU.
        # This avoids potential OOM on GPU for large models that
        # load parameters onto GPU then make a new GPU copy into the parameter
        # buffer. The GPU transfer happens implicitly in load_state_dict.
        model_state = torch.load(model_path, map_location=util.device_mapping(-1))
        training_state = torch.load(training_state_path, map_location=util.device_mapping(-1))
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(training_state["optimizer"])
        if self._learning_rate_scheduler is not None and "learning_rate_scheduler" in training_state:
            self._learning_rate_scheduler.lr_scheduler.load_state_dict(
                    training_state["learning_rate_scheduler"])
        move_optimizer_to_cuda(self.optimizer)

        # We didn't used to save `validation_metric_per_epoch`, so we can't assume
        # that it's part of the trainer state. If it's not there, an empty list is all
        # we can do.
        if "val_metric_per_epoch" not in training_state:
            logger.warning("trainer state `val_metric_per_epoch` not found, using empty list")
            val_metric_per_epoch: List[float] = []
        else:
            val_metric_per_epoch = training_state["val_metric_per_epoch"]

        if isinstance(training_state["epoch"], int):
            epoch_to_return = training_state["epoch"] + 1
        else:
            epoch_to_return = int(training_state["epoch"].split('.')[0]) + 1

        # For older checkpoints with batch_num_total missing, default to old behavior where
        # it is unchanged.
        batch_num_total = training_state.get('batch_num_total')
        if batch_num_total is not None:
            self._batch_num_total = batch_num_total

        return epoch_to_return, val_metric_per_epoch

    # Requires custom from_params.
    @classmethod
    def from_params(cls,  # type: ignore
                    model: Model,
                    serialization_dir: str,
                    iterator: DataIterator,
                    train_data: Iterable[Instance],
                    validation_data: Optional[Iterable[Instance]],
                    params: Params,
                    validation_iterator: DataIterator = None,
                    held_out_train_data: Optional[Iterable[Instance]] = None,
                    held_out_iterator: DataIterator = None,
                    ensemble_model: Model = None) -> 'Trainer':
        # pylint: disable=arguments-differ
        patience = params.pop_int("patience", None)
        validation_metric = params.pop("validation_metric", "-loss")
        shuffle = params.pop_bool("shuffle", True)
        num_epochs = params.pop_int("num_epochs", 20)
        cuda_device = params.pop("cuda_device", -1)
        grad_norm = params.pop_float("grad_norm", None)
        grad_clipping = params.pop_float("grad_clipping", None)
        lr_scheduler_params = params.pop("learning_rate_scheduler", None)

        optimizer_params = params.pop("optimizer")
        if ensemble_model:
            parameters = [[[n, p] for n, p in m.named_parameters() if p.requires_grad]
                          for m in ensemble_model.submodels]
            ensemble_optimizer = [Optimizer.from_params(parameters[i], optimizer_params.duplicate())
                                  for i in range(len(ensemble_model.submodels))]
            optimizer = ensemble_optimizer[0]
        else:
            parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
            optimizer = Optimizer.from_params(parameters, optimizer_params)
            ensemble_optimizer = None

        if lr_scheduler_params:
            if ensemble_model:
                ensemble_scheduler = [LearningRateScheduler.from_params(ensemble_optimizer[i], lr_scheduler_params.duplicate())
                                      for i in range(len(ensemble_model.submodels))]
                scheduler = ensemble_scheduler[0]
            else:
                scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params)
                ensemble_scheduler = None
        else:
            scheduler = None
            ensemble_scheduler = None

        num_serialized_models_to_keep = params.pop_int("num_serialized_models_to_keep", 20)
        keep_serialized_model_every_num_seconds = params.pop_int(
                "keep_serialized_model_every_num_seconds", None)
        model_save_interval = params.pop_float("model_save_interval", None)
        summary_interval = params.pop_int("summary_interval", 100)
        histogram_interval = params.pop_int("histogram_interval", None)
        should_log_parameter_statistics = params.pop_bool("should_log_parameter_statistics", True)
        should_log_learning_rate = params.pop_bool("should_log_learning_rate", False)

        active_learning = params.pop("active_learning", None)

        params.assert_empty(cls.__name__)

        return cls(model, optimizer, iterator,
                   train_data, held_out_train_data, validation_data,
                   patience=patience,
                   validation_metric=validation_metric,
                   validation_iterator=validation_iterator,
                   held_out_iterator=held_out_iterator,
                   shuffle=shuffle,
                   num_epochs=num_epochs,
                   serialization_dir=serialization_dir,
                   cuda_device=cuda_device,
                   grad_norm=grad_norm,
                   grad_clipping=grad_clipping,
                   learning_rate_scheduler=scheduler,
                   num_serialized_models_to_keep=num_serialized_models_to_keep,
                   keep_serialized_model_every_num_seconds=keep_serialized_model_every_num_seconds,
                   model_save_interval=model_save_interval,
                   summary_interval=summary_interval,
                   histogram_interval=histogram_interval,
                   should_log_parameter_statistics=should_log_parameter_statistics,
                   should_log_learning_rate=should_log_learning_rate,
                   active_learning=active_learning,
                   ensemble_model=ensemble_model,
                   ensemble_optimizer=ensemble_optimizer,
                   ensemble_scheduler=ensemble_scheduler)

