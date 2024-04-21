import torch
import numpy as np
import time
import copy

from flcore.clients.clientbase import Client
from collections import OrderedDict
from copy import deepcopy

from typing import Dict, List, Tuple, Union
from typing import Optional, Type

import torch.nn as nn

class clientFedAGHN(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.args = args
        self.before_test_result = 0.0
        self.before_test_acc_num = 0.0
        self.before_test_sample_num = 0.0


    def train(self, new_parameters, return_diff = True):
        trainloader = self.load_train_data()
        # load personalized initial model
        self.model.load_state_dict(new_parameters, strict=False)
        # evaluate personalized initial model
        midtest_acc, midtest_num, _ = self.test_metrics()
        self.before_test_result = midtest_acc * 1.0 / midtest_num
        self.before_test_acc_num = midtest_acc
        self.before_test_sample_num = midtest_num

        # begin local training
        self.model.train()
        start_time = time.time()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        # new return format
        if return_diff and self.args.get_model_param:
            # return delta and param
            delta = OrderedDict()
            for (name, p0), p1 in zip(
                new_parameters.items(), trainable_params(self.model)
            ):
                delta[name] = p0 - p1

            model_param = OrderedDict()
            for (name, p0), p1 in zip(
                new_parameters.items(), trainable_params(self.model, detach=True)
            ):
                model_param[name] = p1

            return delta, model_param
        elif return_diff:
            delta = OrderedDict()
            for (name, p0), p1 in zip(
                new_parameters.items(), trainable_params(self.model)
            ):
                delta[name] = p0 - p1

            return delta
        else:
            return trainable_params(self.model, detach=True)


    def train_metrics(self, model=None):
        trainloader = self.load_train_data()
        if model == None:
            model = self.model
        model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num


# get trainable parameters
def trainable_params(
    src,
    detach=False,
    requires_name=False,
) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[str]]]:

    func = (lambda x: x.detach().clone()) if detach else (lambda x: x)
    parameters = []
    keys = []
    if isinstance(src, OrderedDict):
        for name, param in src.items():
            if param.requires_grad:
                parameters.append(func(param))
                keys.append(name)
    elif isinstance(src, torch.nn.Module):
        for name, param in src.state_dict(keep_vars=True).items():
            if param.requires_grad:
                parameters.append(func(param))
                keys.append(name)

    if requires_name:
        return parameters, keys
    else:
        return parameters

