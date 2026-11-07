import copy
import torch


def average_models(client_updates):
    avg_state = copy.deepcopy(client_updates[0])

    for key in avg_state.keys():

        # 共有層だけの部分更新に対応。BatchNormの整数カウンタを別処理。
        if not torch.is_floating_point(avg_state[key]):
            avg_state[key] = torch.stack([
                update[key] for update in client_updates
            ]).max(dim=0).values
            continue

        for i in range(1, len(client_updates)):
            avg_state[key] += client_updates[i][key]

        avg_state[key] = torch.div(avg_state[key], len(client_updates))
    return avg_state


def federated_learning(client_updates, global_model):
    avg_state = average_models(client_updates)
    global_model.load_state_dict(avg_state, strict=False)
    
    return global_model