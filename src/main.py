import copy
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from .models import EENet20
from .dataset import get_datasets, split_dataset, split_client_tests
from .utils import (
    set_seed, choose_clients, select_optimizer, 
    load_config, acc_average, evaluate_clients,
)
from .trainer import train, test
from .server import federated_learning


def full_state(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def shared_state(model, prefixes):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key.startswith(prefixes)
    }


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))   # FedAvg/src
    parent_dir = os.path.dirname(current_dir)                  # FedAvg
    sys.path.append(parent_dir)

    config = load_config(parent_dir)

    batch_size    = config['training']['batch_size']
    epochs        = config['training']['epochs']
    lr            = config['training']['lr']
    optimizer     = config['training']['optimizer']
    model_name    = config['training']['model_name']
    dataset_name  = config['training']['dataset_name']


    global_rounds = config['FL']['global_rounds']
    num_clients   = config['FL']['num_clients']
    dirichlet     = config['FL']['dirichlet']
    ite_num       = config['FL']['ite_num']
    cohort        = config['FL']['cohort']
    mode          = config['FL']['mode']

    shared_prefixes = ("conv1.","bn1.","layer1.","layer2.0.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset, test_loader = get_datasets(batch_size, dataset_name)

    for ite in range(ite_num):
        seed = 100 + ite
        set_seed(seed)

        global_model = EENet20(channels=(16, 16, 32)).to(device)
        initial_state = full_state(global_model)

        subsets = split_dataset(train_dataset, num_clients, dirichlet)

        test_subsets, train_counts, test_counts = (
            split_client_tests(subsets, test_loader.dataset, seed=seed, num_classes=10, mode=mode)
            )
        
        client_loaders = [
                    DataLoader(
                        subset, batch_size=batch_size, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(seed + i),
                    )
                    for i, subset in enumerate(subsets)
                ]
        
        test_loaders = [
            DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                generator=torch.Generator().manual_seed(seed + i)
            ) for i, subset in enumerate(test_subsets)
        ]
        
        global_model.train()

        acc_history = [[] for _ in range(global_rounds)]
        pFL_acc_history = []
        initial_state = full_state(global_model)

        print(f"iteration {ite+1}/ {ite_num}")

        set_seed(seed)
        selected_clients = [choose_clients(num_clients, cohort) for _ in range(global_rounds)]
        global_model.load_state_dict(initial_state)
        client_states = [full_state(global_model) for _ in range(num_clients)]

        for global_round in range(global_rounds):
            set_seed(seed + global_round)

            print(f"Round {global_round+1}/ {global_rounds}")

            local_models = []            

            for client_idx in selected_clients[global_round]:
                client_loaders[client_idx].generator.manual_seed(seed + global_round + client_idx)

                local_model = copy.deepcopy(global_model)
                local_model.load_state_dict(client_states[client_idx])
                # 共有部分のパラメータをグローバルモデルからロード
                local_model.load_state_dict(
                    shared_state(global_model, shared_prefixes), strict=False
                )
                local_optimizer = select_optimizer(
                    optimizer,
                    local_model,
                    lr
                    )

                # ローカルでモデルを訓練
                for _ in range(epochs):
                    train(local_optimizer, local_model, client_loaders[client_idx], device, model_name)

                client_states[client_idx] = full_state(local_model)
                local_models.append(shared_state(local_model, shared_prefixes))

            # クライアントのモデルを平均化してグローバルモデルを更新
            federated_learning(local_models, global_model)

            test_acc = evaluate_clients(global_model, client_states, test_loaders, device, [10]*num_clients, shared_prefixes)
            acc_history[global_round].append(np.mean(test_acc))
            
            print(f"accuracy: {acc_history[global_round][-1]:.4f}")
        
        # 全ラウンド終了後、共有層を固定して各端末の個人層を1エポックだけ学習
        for client_idx in range(num_clients):
            local_model = copy.deepcopy(global_model)
            local_model.load_state_dict(client_states[client_idx])
            local_model.load_state_dict(
                shared_state(global_model, shared_prefixes), strict=False
            )

            for name, parameter in local_model.named_parameters():
                parameter.requires_grad_(not name.startswith(shared_prefixes))

            local_optimizer = select_optimizer(optimizer, local_model, lr)

            train(
                local_optimizer, local_model, client_loaders[client_idx],
                device, model_name, frozen_layers=shared_prefixes,
            )
            client_states[client_idx] = full_state(local_model)
        
        pFL_acc_history.append(evaluate_clients(
            global_model, client_states, test_loaders, device, [10]*num_clients, shared_prefixes,
        ))
    
    final_acc = acc_average(acc_history)

    # 出力先のパス設定
    save_dir = os.path.join(
        parent_dir,
        "results",
        f"{dataset_name}_"
        f"{model_name}_"
        f"{dirichlet}_,{cohort}/{num_clients}"
        )

    os.makedirs(save_dir, exist_ok=True)

    # csv出力
    df = pd.DataFrame({
        "round": np.arange(1, global_rounds + 1),
        "accuracy": final_acc
    })
    csv_path = os.path.join(save_dir, "FedAvg.csv")
    df.to_csv(csv_path, index=False)

    #グラフ出力
    plt.figure(figsize=(10, 6))

    rounds = np.arange(1, global_rounds + 1) # グラフのX軸（ラウンド数）

    plt.plot(rounds, final_acc, linestyle='-', color='b', label='Simple Average')

    # グラフのタイトルとラベル
    plt.title('FedAvg', fontsize=16)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True) # グリッド線を表示
    plt.legend() # 凡例

    # 範囲指定
    plt.xlim(0, global_rounds)
    plt.ylim(0, 100)

    combined_path = os.path.join(save_dir, "FedAvg.png")

    plt.savefig(combined_path)
        