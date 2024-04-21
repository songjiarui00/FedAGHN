import torch
import copy
import time
import numpy as np
import math
from flcore.clients.clientfedaghn import clientFedAGHN
from flcore.servers.serverbase import Server
from typing import Dict, List, Optional, Type
from typing import Tuple, Union
from collections import OrderedDict
from copy import deepcopy

import os
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
# path for saving results
PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
OUT_DIR = PROJECT_DIR / "out"
TEMP_DIR = PROJECT_DIR / "temp"

import datetime
# get current time
current_time = datetime.datetime.now()
# format as string
TIME_STRING = current_time.strftime("%Y-%m-%d-%H-%M-%S")

# calculate cosine similarity for matrix
def cal_cos_sim_matrix(matrix):
    normalized_matrix = matrix / torch.norm(matrix, dim=1, keepdim=True)
    similarity = torch.matmul(normalized_matrix, normalized_matrix.t())
    return similarity

# calculate cosine similarity matrices for all layers
def cal_cos_sim_matrix_list_all_models(matrix,value_index_list):
    similarity_list = []
    # value_index_list and matrix have initialized value to match the format in the first round
    for i in range(0,len(value_index_list)):
        if len(similarity_list) == 0:
            similarity_array = matrix[:, 0:int(value_index_list[i])]
            firstcosmatrix = cal_cos_sim_matrix(similarity_array)
            similarity_list.append(firstcosmatrix)
        else:
            similarity_array = matrix[:, int(value_index_list[i-1]):int(value_index_list[i])]
            cosmatrix = cal_cos_sim_matrix(similarity_array)
            similarity_list.append(cosmatrix)

    return similarity_list

# get layers and corresponding indexs
def key_value_pairs(keys,origin_valueindex):
    new_valueindex=[]
    new_keys=[]
    for i in range(0,len(keys)-1,1):
        if i+1==len(keys)-1:
            new_valueindex.append(origin_valueindex[i+1])
            if keys[i+1].endswith(".weight"):
                new_keys.append(keys[i+1][:-7])
            elif keys[i+1].endswith(".bias"):
                new_keys.append(keys[i+1][:-5])
        else:
            if keys[i].endswith(".weight") and keys[i + 1].endswith(".bias") and keys[i][:-7] == keys[i + 1][:-5]:
                continue
            elif keys[i].endswith(".bias") and keys[i + 1].endswith(".weight") and keys[i + 1][:-7] != keys[i][:-5]:
                new_valueindex.append(origin_valueindex[i])
                new_keys.append(keys[i][:-5])
            elif keys[i].endswith(".weight") and keys[i + 1].endswith(".weight") and keys[i][:-7] != keys[i + 1][:-7]:
                new_valueindex.append(origin_valueindex[i])
                new_keys.append(keys[i][:-7])

    return new_keys,new_valueindex

# turn state_dic into a vector
def sd_matrixing(state_dic):
    keys = []
    param_vector = None
    value_len_list = []
    value_index_list = []
    len_cnt = 0
    for key, param in state_dic.items():
        keys.append(key)
        if param_vector is None:
            param_vector = param.clone().detach().flatten().cpu()
        else:
            if len(list(param.size())) == 0:
                param_vector = torch.cat((param_vector, param.clone().detach().view(1).cpu().type(torch.float32)), 0)
            else:
                param_vector = torch.cat((param_vector, param.clone().detach().flatten().cpu()), 0)
        # len_cnt denotes the current whole length of parameters
        len_cnt=len_cnt+len(param.clone().detach().flatten().cpu())
        # value_len_list denotes the length of flattened parameters for key
        value_len_list.append(len(param.clone().detach().flatten().cpu()))
        # value_index_list denotes indexs of value arrays
        value_index_list.append(len_cnt)

    return param_vector,value_len_list,value_index_list,keys

# format tansfer
def transfer_delta(input_dict):
    ordered_dict = OrderedDict(sorted(input_dict.items()))
    tensor_list = list(ordered_dict.values())
    combined_tensor = torch.stack(tensor_list)
    return combined_tensor


class FedAGHN(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientFedAGHN)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.Budget = []

        # get model for FedAGHN
        self.model = copy.deepcopy(args.model)
        # select clients
        self.selected_clients = self.select_clients()
        # trainable parameters of client
        self.client_trainable_params = [
            trainable_params(client.model) for client in self.selected_clients
        ]
        self.unique_model = True
        random_init_params, self.trainable_params_name = trainable_params(
            self.model, detach=True, requires_name=True
        )

        # Attentive Graph HyperNetwork
        self.hypernet = HyperNetwork_GHN(
            client_num=self.num_clients,
            backbone=self.model,
            device=self.device,
            beta=self.args.beta_hn,
            gama=self.args.gama_hn,
            gama_position=self.args.gama_position,
            gama_threshold=self.args.gama_threshold,
            gama_outer_policy=self.args.gama_outer_policy,
            gama_modify=self.args.gama_modify,
            beta_modify=self.args.beta_modify,
            gama_classifer_decouple=self.args.gama_classifer_decouple,
            gama_hn_for_classifer=self.args.gama_hn_for_classifer,
        )
        # optimizer for hn
        self.hn_optimizer = torch.optim.SGD(
            self.hypernet.parameters(),
            lr=self.args.hn_lr,
            momentum=self.args.hn_momentum,
        )
        self.test_flag = False
        self.layers_name = [
            name
            for name, module in self.model.named_modules()
            if isinstance(module, nn.Conv2d)
               or isinstance(module, nn.Linear)
               or isinstance(module, nn.BatchNorm2d)
        ]

        # deltalist is used to cache the updated delta_theta
        self.delta_list = {}
        # old_delta_list_embed is used to cache the updated delta_theta in the previous round for calculating
        self.old_delta_list_embed = {}

        # client join rate = 1
        # initialize self.delta_list for the following calculation in the first round
        for client_id in range(0, self.num_clients):
            self.delta_list[client_id] = torch.ones(100000)

        self.value_len_list = None
        # initialize self.value_index_list for the following calculation in the first round
        self.value_index_list = [9 + 9 * i for i in range(2 * len(self.layers_name))]

        self.model_state_dict_keys = [key for key, value in self.model.named_parameters()]
        self.keys = self.model_state_dict_keys
        self.visual_a = args.visual_a


    def train(self):
        # save results for personalized model
        self.rs_test_acc_after = []
        self.rs_train_acc_after = []
        self.rs_val_acc_after = []
        # save weights on graphs during FL process
        final_a_list = []
        # FL process
        for i in range(self.global_rounds+1):
            s_t = time.time()
            # preparation for AGHN's input
            # flatten delta_theta
            flattened_delta = transfer_delta(self.delta_list)
            #print("size of flattened delta", flattened_delta.size()[1])

            print("----------------------pre-calculate cosine similarity matrix------------------------")
            # pre-calculate cosine similarity matrix to save the space on GPUs
            # 1.align named parameters with layers' names by keys
            _, self.value_index_list = key_value_pairs(self.keys, self.value_index_list)
            # 2.calculate cosine similarity matrices for all layers
            self.sim_matrix_layerwise = cal_cos_sim_matrix_list_all_models(flattened_delta.to(self.device),self.value_index_list)
            # * note that in order to save space on GPUs, we pre-calculate cosine similarity matrices, which are indexed in AGHNs
            # 3. (optional) take delta_theta as some features, but they are not used in calculation to save space on GPUs
            self.old_delta_list_embed = flattened_delta[:, 0:10000].to(self.device)
            print("---------------------pre-calculating finished! Begin aggregating------------------")

            # save results of personalized initial model
            before_acc_list = []
            before_acc_num_list = []
            before_sample_num_list = []

            # self.client_trainable_params_last_rounds is used to cache theta (personalized models' parameters) in the previous round
            self.client_trainable_params_last_rounds = copy.deepcopy(self.client_trainable_params)
            final_a = {}

            for client in self.selected_clients:
                # employ AGHN to generate personalized initial model for each client
                client_local_params, normalized_a_layer = self.generate_client_params(client.id, self.old_delta_list_embed, self.sim_matrix_layerwise)
                # local training
                delta, model_param = client.train(client_local_params, return_diff=True)
                # save the layer-wise weights on graphs
                final_a[client.id] = normalized_a_layer
                # save the evaluation of personalized initial model
                before_acc_list.append(client.before_test_result)
                before_acc_num_list.append(client.before_test_acc_num)
                before_sample_num_list.append(client.before_test_sample_num)

                # take delta_theta as node features (self.args.use_cal_embed == "delta")
                transfered_delta, self.value_len_list, self.value_index_list, self.keys = sd_matrixing(delta)
                # cache delta
                self.delta_list[client.id] = transfered_delta

                # update AGHN
                self.update_hn(client.id, delta)
                # calculate theta^(t+1)
                self.update_client_params(client.id, delta)

            final_a_list.append(final_a)

            # evaluate personalized initial models
            test_acc = np.mean(before_acc_list)
            self.rs_test_acc.append(test_acc)
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized initial models")
                print("test_acc:", test_acc)

            # evaluate personalized models
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models after local training")
                self.evaluate(self.rs_test_acc_after, self.rs_train_acc_after, val=self.rs_val_acc_after)

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break


        """
        print("\nBest test accuracy for after local train.")
        print(max(self.rs_test_acc_after))
        print("\ntest accuracy after local train.")
        print(self.rs_test_acc_after)
        """
        print("\nBest val accuracy of personalized models after local training.")
        print(max(self.rs_val_acc_after))
        print("\nval accuracy of personalized models after local training.")
        print(self.rs_val_acc_after)
        best_val_index = np.argmax(np.array(self.rs_val_acc_after))
        print("best test accuracy for after local train",max(self.rs_test_acc_after))
        print("best_val_rounds_index",best_val_index)
        print("best_val_acc",self.rs_val_acc_after[best_val_index])
        # personalized models' best_val_model_test_acc
        print("best_val_model_test_acc", self.rs_test_acc_after[best_val_index])

        # visualization
        """
        # add visualize
        best_a = final_a_list[best_val_index]
        import json
        # best_a
        with open(self.visual_a + '.json', 'w') as f:
            json.dump(best_a, f)
        f.close()
        print("client1", best_a[1])
        """
        """
        for i in range(0,best_val_index):
            temp_a = final_a_list[i]
            with open('visual_time/'+self.visual_a + '_round_' + str(i) + '.json', 'w') as f:
                json.dump(temp_a, f)
            f.close()
        """

        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        print("all results saved")
        self.hypernet.clean()
        print("hypernet cleaned")


    @torch.no_grad()
    def update_client_params(self, client_id, delta):
        new_params = []
        for param, diff in zip(
            self.client_trainable_params[client_id], trainable_params(delta)
        ):
            new_params.append((param - diff.to(self.device)).detach())
        self.client_trainable_params[client_id] = new_params

    # get personalized initial model
    def generate_client_params(self, client_id: int, delta_tensor, sim_matrix):
        aggregated_params = OrderedDict(
            zip(self.trainable_params_name, self.client_trainable_params_last_rounds[client_id])
        )
        if not self.test_flag:
            layer_params_dict = dict(
                zip(
                    self.trainable_params_name, list(zip(*self.client_trainable_params_last_rounds))
                )
            )
            # get weights on graphs
            alpha = self.hypernet(client_id,delta_tensor,sim_matrix)
            #print("alpha",alpha)
            default_weight = torch.zeros(
                self.num_clients, dtype=torch.float, device=self.device
            )
            default_weight[client_id] = 1.0
            normalized_a_layer = []

            for name, params in layer_params_dict.items():
                a = alpha[".".join(name.split(".")[:-1])]
                if a.sum() == 0:
                    a = default_weight

                # aggregate over graphs
                aggregated_params[name] = torch.sum(
                    (a / a.sum()) * torch.stack(params, dim=-1).to(self.device), dim=-1
                )

                normal_a = a / a.sum()
                normalized_a_layer.append(normal_a.cpu().detach().tolist())

            self.client_trainable_params[client_id] = list(aggregated_params.values())
        return aggregated_params, normalized_a_layer


    def update_hn(self, client_id: int, delta) -> None:
        # calculate gradients
        self.hn_optimizer.zero_grad()
        hn_grads = torch.autograd.grad(
            outputs=self.client_trainable_params[client_id],
            inputs=self.hypernet.parameters(),
            grad_outputs=list(
                map(lambda diff: (-diff).clone().detach(), list(delta.values()))
            ),
            allow_unused=True,
        )

        for (name,param), grad in zip(self.hypernet.named_parameters(), hn_grads):
            if grad is not None:
                #print("client_id:"+str(client_id)+" modified param grad:", name)
                #print("client_id:"+str(client_id)+" modified param grad value:", grad)
                param.grad = grad
            else:
                print("client_id:"+str(client_id)+" NOT modified param:", name)

        self.hn_optimizer.step()
        self.hypernet.save_hn()


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



# AGHN
from typing import Optional
import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
import torch_sparse

class AGHNConv(MessagePassing):

    def __init__(self, requires_grad: bool = True, add_self_loops: bool = True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.requires_grad = requires_grad
        self.add_self_loops = add_self_loops
        # two trainable parameters in AGHN are shown in class:HyperNetwork_GHN

    def forward(self, x: Tensor, edge_index: Adj, sim_matrix) -> Tensor:
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index = edge_index

            elif isinstance(edge_index, SparseTensor):
                edge_index = torch_sparse.set_diag(edge_index)

        x_norm = F.normalize(x, p=2., dim=-1)
        self.cos_matrix = sim_matrix
        self.edge_index = edge_index
        # propagate_type: (x: Tensor, x_norm: Tensor)
        return self.propagate(edge_index, x=x, x_norm=x_norm, size=None)

    def message(self, x_j: Tensor, x_norm_i: Tensor, x_norm_j: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # get the index of current node
        current_node = index[0].clone().detach()
        print("current_node", current_node)

        # get current node's index in cosine similarity matrices
        cos_for_current_client_temp = [t[current_node].cpu().numpy().tolist() for t in self.cos_matrix]
        #print("cos_for_current_client_temp",cos_for_current_client_temp)
        cos_for_current_client = torch.tensor(cos_for_current_client_temp).clone().detach()

        # calculate alpha_ij where j!=i
        del_self_cos = torch.cat((cos_for_current_client[:, 0:current_node], cos_for_current_client[:, current_node + 1:]), dim=1)
        scaled_cos_for_current_client = self.beta.view(-1, 1) * del_self_cos.to(self.beta.device)
        #print("scaled_cos_for_current_client", scaled_cos_for_current_client)
        alpha_cos_softmax = torch.nn.functional.softmax(scaled_cos_for_current_client, dim=1)
        #print("alpha_cos_softmax", alpha_cos_softmax)
        self_weighted = self.gama.view(-1, 1).to(alpha_cos_softmax.device)

        # clamp p_i
        if self.gama_outer_policy == "outer_clamp":
            self_weighted = torch.clamp(self_weighted, min=self.gama_threshold)
            self.gama.data = torch.clamp(self.gama.data, min=self.gama_threshold)

        alpha_outer = torch.cat((alpha_cos_softmax[:, 0:current_node], self_weighted, alpha_cos_softmax[:, current_node:]), dim=1)

        self.alpha = alpha_outer
        return x_j


# AGHN
class HyperNetwork_GHN(nn.Module):
    def __init__(
        self,
        client_num: int,
        backbone: nn.Module,
        device: torch.device,
        beta: float,
        gama: float,
        gama_position: str,
        gama_threshold: float,
        gama_outer_policy: str,
        gama_modify: str,
        beta_modify: str,
        gama_classifer_decouple: str,
        gama_hn_for_classifer: float,
    ):
        super(HyperNetwork_GHN, self).__init__()

        self.client_num = client_num
        self.device = device

        # for tracking the current client's hn parameters
        self.client_id: int = None
        self.cache_dir = TEMP_DIR / "FedAGHN_hn_weight" / TIME_STRING
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        self.cache_dir_for_hyperparam = "hyperparam" + '/' + TIME_STRING
        if not os.path.isdir(self.cache_dir_for_hyperparam):
            os.makedirs(self.cache_dir_for_hyperparam, exist_ok=True)

        if gama_position=="outer":
            self.aghnconv = AGHNConv().to(self.device)

        self.layers_name = [
            name
            for name, module in backbone.named_modules()
            if isinstance(module, nn.Conv2d)
            or isinstance(module, nn.Linear)
            or isinstance(module, nn.BatchNorm2d)
        ]

        self.edge_attr=None
        self.aghnconv.gama_threshold = gama_threshold
        self.aghnconv.gama_outer_policy = gama_outer_policy

        # fixed or not
        self.gama_modify = gama_modify
        self.beta_modify = beta_modify
        # set the value
        self.gama_fixed_value = gama
        self.beta_fixed_value = beta
        # if decouple
        self.gama_classifer_decouple = gama_classifer_decouple
        self.gama_hn_for_classifer = gama_hn_for_classifer


        ###初始化的时候保存两个参数：beta和gama
        if os.listdir(self.cache_dir) == []:
            for client_id in range(client_num):
                # initialize trainable parameters in AGHN
                self.aghnconv.beta = Parameter(torch.empty(len(self.layers_name)))
                self.aghnconv.gama = Parameter(torch.empty(len(self.layers_name)))
                self.aghnconv.beta.data.fill_(beta)
                self.aghnconv.gama.data.fill_(gama)
                # add classifier decouple
                if self.gama_classifer_decouple == "decouple":
                    self.aghnconv.gama.data[-1] = self.gama_hn_for_classifer

                # save aghn
                torch.save(
                    {
                        "aghn": deepcopy(self.aghnconv.state_dict()),
                    },
                    self.cache_dir / f"{client_id}.pt",
                )

    def forward(self, client_id: int,delta_tensor,sim_matrix):
        self.client_id = client_id
        edge_index = self.create_edge_index(self.client_id, self.client_num)
        self.load_hn()
        x = self.aghnconv(delta_tensor.to(self.device), edge_index.to(self.device),sim_matrix)
        edge_weights=self.aghnconv.alpha.clone()
        edge_index=self.aghnconv.edge_index.clone()
        self.edge_attr = edge_weights
        alpha = edge_weights
        #print("len of layersname", len(self.layers_name))
        return {layer: a for layer, a in zip(self.layers_name, alpha)}

    def create_edge_index(self,client_id, client_num):
        out_edges = torch.arange(client_num)
        in_edges = torch.tensor([client_id] * client_num)
        edge_index = torch.stack([out_edges, in_edges], dim=0)
        return edge_index

    def save_hn(self):
        torch.save(
            {
                "aghn": deepcopy(self.aghnconv.state_dict()),
                # "edge_attr":self.edge_attr
            },
            self.cache_dir / f"{self.client_id}.pt",
        )
        print("save dict", self.aghnconv.state_dict())
        print(f"access save aghn for {self.client_id}")
        self.client_id = None

    def load_hn(self):
        weights = torch.load(self.cache_dir / f"{self.client_id}.pt")
        self.aghnconv.load_state_dict(weights["aghn"])

        if self.gama_modify == "fixed":
            self.aghnconv.gama.data.fill_(self.gama_fixed_value)
            print("gama is fixed!")
            # add classifier decouple
            if self.gama_classifer_decouple == "decouple":
                self.aghnconv.gama.data[-1] = self.gama_hn_for_classifer
        if self.beta_modify == "fixed":
            self.aghnconv.beta.data.fill_(self.beta_fixed_value)
            print("beta is fixed!")

        self.aghnconv.to(self.device)
        # self.edge_attr=weights["edge_attr"]
        print(f"access load aghn for {self.client_id}")

    def clean(self):
        if os.path.isdir(self.cache_dir):
            os.system(f"rm -rf {self.cache_dir}")


