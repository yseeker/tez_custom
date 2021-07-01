"""
The tez model class
"""

import warnings
import os
import random
import psutil
import torch
import torch.nn as nn
import numpy as np
from tez import enums
from tez.callbacks import CallbackRunner
from tez.utils import AverageMeter
#from tqdm import tqdm
from tqdm.notebook import tqdm
import wandb
import gc
#from tqdm.notebook import tqdm

#warnings.filterwarnings("ignore", message=torch.optim.lr_scheduler.SAVE_STATE_WARNING)
warnings.filterwarnings("ignore")

def set_seed(seed = 42):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    return random_state

def mixup_data(inputs, targets, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = inputs.size()[0]
    index = torch.randperm(batch_size)
    mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
    targets_a, targets_b = targets, targets[index]
    
    return mixed_inputs, targets_a, targets_b, lam

def mixup_criterion(criterion, outputs, targets_a, targets_b, lam):
    return lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)

class Trainer():
    def __init__(self, *args, **kwrds):
        self.model = None
        self.valid_targets = None
        self.train_loader = None
        self.valid_loader = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.step_scheduler_after = None
        self.step_scheduler_metric = None
        self.current_epoch = 0
        self.current_train_step = 0
        self.current_valid_step = 0
        self._model_state = None
        self._train_state = None
        self.device = None
        self._callback_runner = None
        self.fp16 = False
        self.scaler = None
        self.metrics = {}
        self.metrics["train"] = {}
        self.metrics["valid"] = {}
        self.metrics["test"] = {}

    @property
    def model_state(self):
        return self._model_state

    @model_state.setter
    def model_state(self, value):
        self._model_state = value

    @property
    def train_state(self):
        return self._train_state

    @train_state.setter
    def train_state(self, value):
        self._train_state = value
        if self._callback_runner is not None:
            self._callback_runner(value)

    def name_to_metric(self, metric_name):
        if metric_name == "current_epoch":
            return self.current_epoch
        v_1 = metric_name.split("_")[0]
        v_2 = "_".join(metric_name.split("_")[1:])
        return self.metrics[v_1][v_2]

    def _init_trainer(
        self,
        device,
        model,
        train_dataset,
        valid_dataset,
        valid_targets,
        train_sampler,
        valid_sampler,
        train_bs,
        valid_bs,
        n_jobs,
        callbacks,
        fp16,
        train_collate_fn,
        valid_collate_fn,
        determinstic,
        benchmark,
    ):

        torch.backends.cudnn.deterministic = determinstic
        torch.backends.cudnn.benchmark = benchmark

        if callbacks is None:
            callbacks = list()

        if n_jobs == -1:
            n_jobs = psutil.cpu_count()

        self.device = device

        if self.valid_targets is None:
            self.valid_targets = valid_targets

        if next(self.model.parameters()).device != self.device:
            self.model.to(self.device)

        if self.train_loader is None:
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=train_bs,
                num_workers=n_jobs,
                sampler=train_sampler,
                shuffle=True,
                collate_fn=train_collate_fn,
                pin_memory = True,
                drop_last = True
            )
        if self.valid_loader is None:
            if valid_dataset is not None:
                self.valid_loader = torch.utils.data.DataLoader(
                    valid_dataset,
                    batch_size=valid_bs,
                    num_workers=n_jobs,
                    sampler=valid_sampler,
                    shuffle=False,
                    collate_fn=valid_collate_fn,
                    pin_memory = True,
                    drop_last = False
                )


        if self.optimizer is None:
            self.optimizer = self.configure_optimizer()

        if self.scheduler is None:
            self.scheduler = self.configure_scheduler()

        self.fp16 = fp16
        if self.fp16:
            self.scaler = torch.cuda.amp.GradScaler()

        self._callback_runner = CallbackRunner(callbacks, self)
        self.train_state = enums.TrainingState.TRAIN_START

    def _init_wandb(self, cfg):
        hyperparams = {
            'batch_size' : cfg.batch_size,
            'n_fold' : cfg.num_of_fold,
            'num_workers' : cfg.num_workers,
            'epochs' : cfg.epochs
        }
        wandb.init(
            config = hyperparams,
            project= cfg.project_name,
            name=cfg.wandb_exp_name,
        )
        wandb.watch(self.model)

    def epoch_metrics(self, *args, **kwargs):
        return

    def monitor_metrics(self, *args, **kwargs):
        return

    def loss(self, *args, **kwargs):
        return

    def configure_optimizer(self, *args, **kwargs):
        return

    def configure_scheduler(self, *args, **kwargs):
        return

    def train_one_step(self, inputs, targets):
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        self.optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            if self.fp16:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets.view(-1, 1))
                    metrics = self.monitor_metrics(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets.view(-1, 1))
                metrics = self.monitor_metrics(outputs, targets)
                loss.backward()
                self.optimizer.step()
            if self.scheduler:
                if self.step_scheduler_after == "batch":
                    if self.step_scheduler_metric is None:
                        self.scheduler.step()
                    else:
                        step_metric = self.name_to_metric(self.step_scheduler_metric)
                        self.scheduler.step(step_metric)
        return outputs, loss, metrics


    def validate_one_step(self, inputs, targets = None):
        inputs = inputs.to(self.device, non_blocking=True)
        if targets is not None:
            targets = targets.to(self.device, non_blocking=True)
            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets.view(-1, 1))
                metrics = self.monitor_metrics(outputs, targets)
            return outputs, loss, metrics
        else:
            outputs = self.model(inputs)
            return outputs, None, None

    def predict_one_step(self, inputs):
        outputs, _, _ = self.validate_one_step(inputs)
        return outputs

    def update_metrics(self, losses, monitor):
        self.metrics[self._model_state.value].update(monitor)
        self.metrics[self._model_state.value]["loss"] = losses.avg

    def train_one_epoch(self, data_loader):
        self.model.train()
        self.model_state = enums.ModelState.TRAIN
        losses = AverageMeter()
        tk0 = tqdm(data_loader, total=len(data_loader), position = 0, leave = True)

        for b_idx, (inputs, targets) in enumerate(tk0):
            self.train_state = enums.TrainingState.TRAIN_STEP_START
            
            _, loss, metrics = self.train_one_step(inputs, targets)
            
            self.train_state = enums.TrainingState.TRAIN_STEP_END
            losses.update(loss.item(), data_loader.batch_size)
            if b_idx == 0:
                metrics_meter = {k: AverageMeter() for k in metrics}
            monitor = {}
            for m_m in metrics_meter:
                if metrics[m_m] is not None:
                    metrics_meter[m_m].update(metrics[m_m], data_loader.batch_size)
                    monitor[m_m] = metrics_meter[m_m].avg
            self.current_train_step += 1
            current_lr = self.optimizer.param_groups[0]['lr']
            wandb_log = {
                "train/step" : b_idx,
                "train/loss_step": losses.avg,
                "lr": current_lr 
                }
            wandb_log.update(monitor)
            wandb.log(wandb_log)
            tk0.set_postfix(loss=losses.avg, stage="train", **monitor, lr = current_lr)
        tk0.close()
        self.update_metrics(losses=losses, monitor=monitor)
        return losses.avg

    def validate_one_epoch(self, data_loader):
        self.model.eval()
        self.model_state = enums.ModelState.VALID
        losses = AverageMeter()
        preds_list = []
        tk0 = tqdm(data_loader, total=len(data_loader))
        for b_idx, (inputs, targets) in enumerate(tk0):
            self.train_state = enums.TrainingState.VALID_STEP_START
            with torch.no_grad():
                output, loss, metrics = self.validate_one_step(inputs, targets)
                preds_list.append(output.cpu().detach().numpy())
            self.train_state = enums.TrainingState.VALID_STEP_END
            losses.update(loss.item(), data_loader.batch_size)
            if b_idx == 0:
                metrics_meter = {k: AverageMeter() for k in metrics}
            monitor = {}
            for m_m in metrics_meter:
                if metrics[m_m] is not None:
                    metrics_meter[m_m].update(metrics[m_m], data_loader.batch_size)
                    monitor[m_m] = metrics_meter[m_m].avg
            tk0.set_postfix(loss=losses.avg, stage="valid", **monitor)
            wandb_log = {
                "valid/step" : b_idx,
                "valid/loss_step": losses.avg,
                }
            wandb_log.update(monitor)
            wandb.log(wandb_log)
            self.current_valid_step += 1
        preds_arr = np.concatenate(preds_list)
        valid_metric_val = self.epoch_metrics(preds_arr, self.valid_targets)
        tk0.close()
        self.update_metrics(losses=losses, monitor=monitor)
        return valid_metric_val, losses.avg

    def process_output(self, output):
        output = output.cpu().detach().numpy()
        return output

    def predict(self, dataset, sampler=None, batch_size=16, n_jobs=1, collate_fn=None):
        if next(self.model.parameters()).device != self.device:
            self.model.to(self.device)

        if n_jobs == -1:
            n_jobs = psutil.cpu_count()

        if batch_size == 1:
            n_jobs = 0
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=n_jobs, sampler=sampler, collate_fn=collate_fn, pin_memory=True
        )

        if self.training:
            self.model.eval()

        preds_list = []
        tk0 = tqdm(data_loader, total=len(data_loader))

        for b_idx, inputs in enumerate(tk0):
            with torch.no_grad():
                preds_one_batch = self.predict_one_step(inputs)
                preds_list.append(preds_one_batch.cpu().detach().numpy())
            tk0.set_postfix(stage="inference")
        tk0.close()
        preds_arr = np.concatenate(preds_list)
        return preds_arr

    def save(self, model_path):
        model_state_dict = self.state_dict()
        if self.optimizer is not None:
            opt_state_dict = self.optimizer.state_dict()
        else:
            opt_state_dict = None
        if self.scheduler is not None:
            sch_state_dict = self.scheduler.state_dict()
        else:
            sch_state_dict = None
        model_dict = {}
        model_dict["state_dict"] = model_state_dict
        model_dict["optimizer"] = opt_state_dict
        model_dict["scheduler"] = sch_state_dict
        model_dict["epoch"] = self.current_epoch
        model_dict["fp16"] = self.fp16
        torch.save(model_dict, model_path)

    def load(self, model_path, device="cuda:0"):
        self.device = device
        if next(self.model.parameters()).device != self.device:
            self.model.to(self.device)
        model_dict = torch.load(model_path, map_location=torch.device(device))
        self.load_state_dict(model_dict["state_dict"])

    def fit(
        self,
        cfg,
        train_dataset,
        valid_dataset=None,
        valid_targets = None,
        train_sampler=None,
        valid_sampler=None,
        device="cuda:0",
        epochs=10,
        train_bs=16,
        valid_bs=16,
        n_jobs=8,
        callbacks=None,
        benchmark = True,
        determinstic = True,
        fp16=False,
        train_collate_fn=None,
        valid_collate_fn=None,
    ):

        set_seed(cfg.seed)
        self._init_trainer(
            device=device,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            valid_targets = valid_targets,
            train_sampler=train_sampler,
            valid_sampler=valid_sampler,
            train_bs=train_bs,
            valid_bs=valid_bs,
            n_jobs=n_jobs,
            callbacks=callbacks,
            fp16=fp16,
            train_collate_fn=train_collate_fn,
            valid_collate_fn=valid_collate_fn,
            determinstic = determinstic,
            benchmark = benchmark,
        )
        self._init_wandb(cfg)

        for epoch in range(epochs):
            self.train_state = enums.TrainingState.EPOCH_START
            self.train_state = enums.TrainingState.TRAIN_EPOCH_START
            train_loss = self.train_one_epoch(self.train_loader)
            self.train_state = enums.TrainingState.TRAIN_EPOCH_END
            if self.valid_loader:
                self.train_state = enums.TrainingState.VALID_EPOCH_START
                valid_metrics, valid_loss = self.validate_one_epoch(self.valid_loader)
                self.train_state = enums.TrainingState.VALID_EPOCH_END
            if self.scheduler:
                if self.step_scheduler_after == "epoch":
                    if self.step_scheduler_metric is None:
                        self.scheduler.step()
                    else:
                        step_metric = self.name_to_metric(self.step_scheduler_metric)
                        self.scheduler.step(step_metric)
            self.train_state = enums.TrainingState.EPOCH_END
            print(f'epoch: {epoch}, epoch_valid_metrics : {valid_metrics}')
            wandb.log({
                "epoch" : epoch,
                "train/loss" : train_loss,
                "valid/loss" : valid_loss,
                "valid/metric" : valid_metrics,
                })
            if self._model_state.value == "end":
                break
            self.current_epoch += 1
        self.train_state = enums.TrainingState.TRAIN_END
        wandb.finish()
        torch.cuda.empty_cache()
        gc.collect()
