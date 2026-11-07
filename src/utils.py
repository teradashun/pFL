import os
import yaml
import copy
import random
import numpy as np
import torch
import torch.optim as optim

from .trainer import test


def load_config(dir):   #引数のパスはFedAvg
    config_path = os.path.join(dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_optimizer(optim_name, local_model, lr):
    if optim_name == "SGD":
        return optim.SGD(filter(lambda p: p.requires_grad, local_model.parameters()), lr=lr)
    
    elif optim_name == "Adam":
        return optim.Adam(filter(lambda p: p.requires_grad, local_model.parameters()), lr=lr)
    
    else:
        raise NameError(f"オプティマイザの名前 '{optim_name}' はサポートされていません")


def choose_clients(num_clients, cohort):
    return random.sample(range(num_clients), cohort)


def acc_average(acc_history: list[list]) -> list:
    acc_history = np.array(acc_history)
    final_acc = np.mean(acc_history, axis=1)
    return final_acc


def evaluate_clients(global_model, client_states, test_loaders, device, evaluation_exits, prefixes):
        current_shared = {
            key: value for key, value in global_model.state_dict().items()
            if key.startswith(prefixes)
        }
        evaluation_model = copy.deepcopy(global_model)
        accuracies = []

        for i in range(len(client_states)):
            evaluation_model.load_state_dict(
                client_states[i],
                strict=True,
            )
            evaluation_model.load_state_dict(
                current_shared,
                strict=False,
            )

            accuracies.append(
                test(
                    evaluation_model,
                    test_loaders[i],
                    device,
                    evaluation_exits[i],
                )
            )

        return accuracies