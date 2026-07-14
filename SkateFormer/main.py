from __future__ import print_function

import argparse
import json
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob

# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from torchlight import DictAction

# LR Scheduler
from timm.scheduler.cosine_lr import CosineLRScheduler

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        if self.reduction == 'none':
            return loss
        if self.reduction == 'sum':
            return loss.sum()
        return loss.mean()


def get_parser():
    parser = argparse.ArgumentParser(description='SkateFormer: Skeletal-Temporal Trnasformer for Human Action Recognition')
    parser.add_argument('--work-dir', default='./work_dir', help='the work folder for storing results')
    parser.add_argument('--model_saved_name', default='')
    parser.add_argument('--config', default='./config', help='path to the configuration file')

    # processor
    parser.add_argument('--phase', default='train', help='must be train or test')
    parser.add_argument('--save-score', type=str2bool, default=False, help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument('--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument('--log-interval', type=int, default=100, help='the interval for printing messages (#iteration)')
    parser.add_argument('--save-interval', type=int, default=1, help='the interval for storing models (#iteration)')
    parser.add_argument('--save-epoch', type=int, default=30, help='the start epoch to save model (#iteration)')
    parser.add_argument('--eval-interval', type=int, default=5, help='the interval for evaluating models (#iteration)')
    parser.add_argument('--print-log', type=str2bool, default=True, help='print logging or not')
    parser.add_argument('--show-topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument('--num-worker', type=int, default=4, help='the number of worker for data loader')
    parser.add_argument('--train-feeder-args', action=DictAction, default=dict(), help='the arguments of data loader for training')
    parser.add_argument('--test-feeder-args', action=DictAction, default=dict(), help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--model-args', action=DictAction, default=dict(), help='the arguments of model')
    parser.add_argument('--weights', default=None, help='the weights for network initialization')
    parser.add_argument('--ignore-weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')
    parser.add_argument('--intent-label-path', default=None, help='path to the coarse-intent mapping json')
    parser.add_argument('--intent-label-key', default=None, help='mapping key inside the coarse-intent json')
    parser.add_argument('--intent-loss-weight', type=float, default=0.0, help='loss weight for auxiliary coarse-intent supervision')
    parser.add_argument('--intent-loss-weight-by-ratio', action=DictAction, default=None,
                        help='per-ratio auxiliary intent weights, e.g. 0.1=1.0 0.3=0.8')
    parser.add_argument('--consistency-loss-weight', type=float, default=0.0,
                        help='loss weight for full-to-prefix consistency KL')
    parser.add_argument('--consistency-temperature', type=float, default=2.0,
                        help='temperature for the consistency KL loss')
    parser.add_argument('--consistency-detach-full', type=str2bool, default=True,
                        help='treat the full-sequence view as a detached teacher target')

    # optim
    parser.add_argument('--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--min-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--warmup-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--warmup_prefix', type=bool, default=False)
    parser.add_argument('--warm_up_epoch', type=int, default=0)
    parser.add_argument('--grad-clip', type=bool, default=False)
    parser.add_argument('--grad-max', type=float, default=1.0)
    parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='AdamW', help='type of optimizer')
    parser.add_argument('--lr-scheduler', default='cosine', help='type of learning rate scheduler')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument('--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument('--start-epoch', type=int, default=0, help='start training from which epoch')
    parser.add_argument('--num-epoch', type=int, default=80, help='stop training in which epoch')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay for optimizer')
    parser.add_argument('--lr-ratio', type=float, default=0.001, help='decay rate for learning rate')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--loss-type', type=str, default='CE')
    return parser


class Processor():
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()

        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')

        self.global_step = 0
        self.load_model()
        self.load_consistency_supervision()
        self.load_intent_supervision()
        self.load_data()

        if self.arg.phase == 'train':
            self.load_optimizer()
            self.load_scheduler(len(self.data_loader['train']))

        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0
        self.model = self.model.to(self.device)

        if self.device.type == 'cuda' and type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{output_device}')
        else:
            self.device = torch.device('cpu')
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        self.model = Model(**self.arg.model_args)
        self.loss = self.build_classification_loss()

        if self.arg.weights:
            #self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights, map_location=self.device)

            weights = OrderedDict([[k.split('module.')[-1], v.to(self.device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))
            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

    def build_classification_loss(self, reduction='mean'):
        if self.arg.loss_type == 'CE':
            return nn.CrossEntropyLoss(reduction=reduction).to(self.device)
        return LabelSmoothingCrossEntropy(smoothing=0.1, reduction=reduction).to(self.device)

    @staticmethod
    def _normalize_ratio_key(ratio):
        return round(float(ratio), 4)

    def _normalize_intent_weight_schedule(self, weight_schedule):
        if not weight_schedule:
            return {}
        normalized = {}
        for ratio, weight in weight_schedule.items():
            normalized[self._normalize_ratio_key(ratio)] = float(weight)
        return normalized

    def load_intent_supervision(self):
        self.intent_loss_weight = float(getattr(self.arg, 'intent_loss_weight', 0.0))
        self.intent_loss_weight_by_ratio = self._normalize_intent_weight_schedule(
            getattr(self.arg, 'intent_loss_weight_by_ratio', None)
        )
        self.intent_label_map = None
        self.intent_loss = None

        if self.intent_loss_weight <= 0 and not self.intent_loss_weight_by_ratio:
            return

        if not self.arg.intent_label_path or not self.arg.intent_label_key:
            raise ValueError('intent-only requires both --intent-label-path and --intent-label-key.')

        with open(self.arg.intent_label_path, 'r') as f:
            mapping_config = json.load(f)

        if self.arg.intent_label_key not in mapping_config:
            raise KeyError(f'Intent mapping key {self.arg.intent_label_key} not found in {self.arg.intent_label_path}.')

        label_map = mapping_config[self.arg.intent_label_key]
        self.intent_label_map = torch.tensor(label_map, dtype=torch.long, device=self.device)
        self.intent_loss = self.build_classification_loss(reduction='none')

        if self.intent_label_map.numel() != getattr(self.model, 'num_classes', self.intent_label_map.numel()):
            raise ValueError(
                f'Intent label map length {self.intent_label_map.numel()} does not match model.num_classes={self.model.num_classes}.'
            )

        expected_intent_classes = int(self.intent_label_map.max().item()) + 1
        model_intent_classes = int(getattr(self.model, 'intent_num_classes', 0))
        if model_intent_classes <= 0:
            raise ValueError('Intent supervision is enabled but model_args.intent_num_classes is not set.')
        if model_intent_classes != expected_intent_classes:
            raise ValueError(
                f'model intent_num_classes={model_intent_classes} does not match mapping classes={expected_intent_classes}.'
            )

    def load_consistency_supervision(self):
        self.consistency_loss_weight = float(getattr(self.arg, 'consistency_loss_weight', 0.0))
        self.consistency_temperature = float(getattr(self.arg, 'consistency_temperature', 2.0))
        self.consistency_detach_full = bool(getattr(self.arg, 'consistency_detach_full', True))
        self.use_consistency = self.consistency_loss_weight > 0.0

        if self.consistency_temperature <= 0:
            raise ValueError('consistency_temperature must be positive.')

    def get_intent_loss_weights(self, observation_ratio, batch_size, device):
        if observation_ratio is None:
            return torch.full((batch_size,), self.intent_loss_weight, dtype=torch.float32, device=device)

        if torch.is_tensor(observation_ratio):
            observation_ratio = observation_ratio.to(device=device, dtype=torch.float32).view(-1)
        else:
            observation_ratio = torch.tensor(observation_ratio, dtype=torch.float32, device=device).view(-1)

        weights = torch.full_like(observation_ratio, self.intent_loss_weight, dtype=torch.float32)
        for ratio, weight in self.intent_loss_weight_by_ratio.items():
            ratio_tensor = torch.full_like(observation_ratio, float(ratio))
            weight_tensor = torch.full_like(observation_ratio, float(weight))
            weights = torch.where(torch.isclose(observation_ratio, ratio_tensor, atol=1e-4, rtol=0.0),
                                  weight_tensor, weights)
        return weights

    def unpack_model_output(self, output):
        if isinstance(output, dict):
            action_output = output.get('action')
            if action_output is None:
                raise ValueError('Model output dict must contain an "action" key.')
            aux_output = {k: v for k, v in output.items() if k != 'action'}
            return action_output, aux_output
        return output, {}

    def unpack_batch(self, batch):
        full_data = None
        full_index_t = None
        if len(batch) == 4:
            data, index_t, label, index = batch
            observation_ratio = None
        elif len(batch) == 5:
            data, index_t, label, index, observation_ratio = batch
        elif len(batch) == 6:
            data, index_t, full_data, full_index_t, label, index = batch
            observation_ratio = None
        elif len(batch) == 7:
            data, index_t, full_data, full_index_t, label, index, observation_ratio = batch
        else:
            raise ValueError(f'Expected batch with 4, 5, 6, or 7 items, got {len(batch)}.')
        return data, index_t, label, index, observation_ratio, full_data, full_index_t

    def compute_supervised_losses(self, output, label, observation_ratio=None, include_intent=True):
        action_output, aux_output = self.unpack_model_output(output)
        total_loss = self.loss(action_output, label)
        loss_stats = {}

        if include_intent and self.intent_loss is not None:
            if 'intent' not in aux_output:
                raise ValueError('Intent supervision is enabled but model output does not provide intent logits.')
            intent_target = self.intent_label_map[label]
            intent_loss = self.intent_loss(aux_output['intent'], intent_target)
            intent_weights = self.get_intent_loss_weights(
                observation_ratio=observation_ratio,
                batch_size=intent_loss.shape[0],
                device=intent_loss.device
            )
            weighted_intent_loss = (intent_loss * intent_weights).mean()
            total_loss = total_loss + weighted_intent_loss
            intent_pred = aux_output['intent'].argmax(dim=1)
            loss_stats['intent_loss'] = intent_loss.mean().detach().item()
            loss_stats['intent_weighted_loss'] = weighted_intent_loss.detach().item()
            loss_stats['intent_weight'] = intent_weights.mean().detach().item()
            loss_stats['intent_acc'] = (intent_pred == intent_target).float().mean().detach().item()

        return action_output, total_loss, loss_stats

    def compute_consistency_loss(self, prefix_logits, full_logits):
        temperature = self.consistency_temperature
        teacher_logits = full_logits.detach() if self.consistency_detach_full else full_logits
        target_probs = F.softmax(teacher_logits / temperature, dim=-1)
        log_probs = F.log_softmax(prefix_logits / temperature, dim=-1)
        return F.kl_div(log_probs, target_probs, reduction='batchmean') * (temperature ** 2)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def load_scheduler(self, n_iter_per_epoch):
        num_steps = int(self.arg.num_epoch * n_iter_per_epoch)
        warmup_steps = int(self.arg.warm_up_epoch * n_iter_per_epoch)

        self.lr_scheduler = None
        if self.arg.lr_scheduler == 'cosine':
            self.lr_scheduler = CosineLRScheduler(
                self.optimizer,
                t_initial=(num_steps - warmup_steps) if self.arg.warmup_prefix else num_steps,
                lr_min=self.arg.min_lr,
                warmup_lr_init=self.arg.warmup_lr,
                warmup_t=warmup_steps,
                cycle_limit=1,
                t_in_epochs=False,
                warmup_prefix=self.arg.warmup_prefix,
            )
        else:
            raise ValueError()

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']

        loss_value = []
        acc_value = []
        prefix_loss_value = []
        consistency_loss_value = []
        intent_loss_value = []
        intent_weighted_loss_value = []
        intent_acc_value = []
        intent_weight_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)

        for batch_idx, batch in enumerate(process):
            data, index_t, label, index, observation_ratio, full_data, full_index_t = self.unpack_batch(batch)
            self.lr_scheduler.step_update(self.global_step)
            self.global_step += 1
            with torch.no_grad():
                data = data.float().to(self.device)
                index_t = index_t.float().to(self.device)
                label = label.long().to(self.device)
                if observation_ratio is not None:
                    observation_ratio = observation_ratio.float().to(self.device)
                if full_data is not None:
                    full_data = full_data.float().to(self.device)
                if full_index_t is not None:
                    full_index_t = full_index_t.float().to(self.device)
            timer['dataloader'] += self.split_time()

            # forward
            model_output = self.model(data, index_t)
            output, prefix_loss, aux_stats = self.compute_supervised_losses(
                model_output, label, observation_ratio=observation_ratio, include_intent=True
            )
            loss = prefix_loss
            prefix_loss_value.append(prefix_loss.detach().item())

            if self.use_consistency:
                if full_data is None or full_index_t is None:
                    raise ValueError(
                        'consistency baseline requires the feeder to return full-sequence views. '
                        'Set train_feeder_args.return_full_sequence=True.'
                    )
                full_model_output = self.model(full_data, full_index_t)
                full_output, _ = self.unpack_model_output(full_model_output)
                consistency_loss = self.compute_consistency_loss(output, full_output)
                loss = prefix_loss + self.consistency_loss_weight * consistency_loss
                aux_stats['consistency_loss'] = consistency_loss.detach().item()
                consistency_loss_value.append(consistency_loss.detach().item())

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            if self.arg.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.arg.grad_max)
            self.optimizer.step()

            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value.append(acc.data.item())
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)
            self.train_writer.add_scalar('prefix_action_loss', prefix_loss.detach().item(), self.global_step)
            if 'consistency_loss' in aux_stats:
                self.train_writer.add_scalar('consistency_loss', aux_stats['consistency_loss'], self.global_step)
            if 'intent_loss' in aux_stats:
                intent_loss_value.append(aux_stats['intent_loss'])
                intent_weighted_loss_value.append(aux_stats['intent_weighted_loss'])
                intent_acc_value.append(aux_stats['intent_acc'])
                intent_weight_value.append(aux_stats['intent_weight'])
                self.train_writer.add_scalar('intent_loss', aux_stats['intent_loss'], self.global_step)
                self.train_writer.add_scalar('intent_weighted_loss', aux_stats['intent_weighted_loss'], self.global_step)
                self.train_writer.add_scalar('intent_acc', aux_stats['intent_acc'], self.global_step)
                self.train_writer.add_scalar('intent_weight', aux_stats['intent_weight'], self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%.'.format(np.mean(loss_value),
                                                                                np.mean(acc_value) * 100))
        self.print_log('\tMean prefix action loss: {:.4f}.'.format(np.mean(prefix_loss_value)))
        if consistency_loss_value:
            self.print_log('\tMean consistency KL loss: {:.4f}.'.format(np.mean(consistency_loss_value)))
        if intent_loss_value:
            self.print_log(
                '\tMean intent loss: {:.4f}.  Mean weighted intent loss: {:.4f}.  Mean intent acc: {:.2f}%.  Mean intent lambda: {:.4f}.'.format(
                    np.mean(intent_loss_value), np.mean(intent_weighted_loss_value),
                    np.mean(intent_acc_value) * 100, np.mean(intent_weight_value)
                )
            )
        self.print_log('\tLearning Rate: {:.4f}'.format(self.lr))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])

            torch.save(weights,
                       self.arg.model_saved_name + '-' + str(epoch + 1) + '-' + str(int(self.global_step)) + '.pt')

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            intent_loss_value = []
            intent_weighted_loss_value = []
            intent_acc_value = []
            intent_weight_value = []
            score_frag = []
            label_list = []
            pred_list = []
            step = 0
            process = tqdm(self.data_loader[ln])
            for batch_idx, batch in enumerate(process):
                data, index_t, label, index, observation_ratio, _full_data, _full_index_t = self.unpack_batch(batch)
                label_list.append(label)
                with torch.no_grad():
                    data = data.float().to(self.device)
                    index_t = index_t.float().to(self.device)
                    label = label.long().to(self.device)
                    if observation_ratio is not None:
                        observation_ratio = observation_ratio.float().to(self.device)
                    model_output = self.model(data, index_t)
                    output, loss, aux_stats = self.compute_supervised_losses(
                        model_output, label, observation_ratio=observation_ratio, include_intent=True
                    )
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())
                    if 'intent_loss' in aux_stats:
                        intent_loss_value.append(aux_stats['intent_loss'])
                        intent_weighted_loss_value.append(aux_stats['intent_weighted_loss'])
                        intent_acc_value.append(aux_stats['intent_acc'])
                        intent_weight_value.append(aux_stats['intent_weight'])

                    _, predict_label = torch.max(output.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())
                    step += 1

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')

            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)
            if 'ucla' in self.arg.feeder:
                self.data_loader[ln].dataset.sample_name = np.arange(len(score))
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1

            print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            if intent_loss_value:
                self.print_log('\tMean {} intent loss: {}.'.format(ln, np.mean(intent_loss_value)))
                self.print_log('\tMean {} weighted intent loss: {}.'.format(ln, np.mean(intent_weighted_loss_value)))
                self.print_log('\tMean {} intent acc: {:.2f}%.'.format(ln, np.mean(intent_acc_value) * 100))
                self.print_log('\tMean {} intent lambda: {:.4f}.'.format(ln, np.mean(intent_weight_value)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

            # acc for each class:
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)
            confusion = confusion_matrix(label_list, pred_list)
            list_diag = np.diag(confusion)
            list_raw_sum = np.sum(confusion, axis=1)
            each_acc = list_diag / list_raw_sum
            with open('{}/epoch{}_{}_each_class_acc.csv'.format(self.arg.work_dir, epoch + 1, ln), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(each_acc)
                writer.writerows(confusion)

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size

            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)

            self.print_log(f'# Parameters: {count_parameters(self.model)}')
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                if epoch + 1 < self.arg.num_epoch * 0.9:
                    self.train(epoch, save_model=False)
                else:
                    self.train(epoch, save_model=True)
                    self.eval(epoch, save_score=True, loader_name=['test'])

            # test the best model
            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-' + str(self.best_acc_epoch) + '*'))[0]
            weights = torch.load(weights_path, map_location=self.device)
            if self.device.type == 'cuda' and type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights = OrderedDict([['module.' + k, v.to(self.device)] for k, v in weights.items()])
            self.model.load_state_dict(weights)

            wf = weights_path.replace('.pt', '_wrong.txt')
            rf = weights_path.replace('.pt', '_right.txt')
            self.arg.print_log = False
            self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.arg.print_log = True

            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')

            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
