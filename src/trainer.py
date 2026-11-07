import torch
import torch.nn as nn


def train(optimizer, model, train_loader, device, model_name, frozen_layers=()):
    # 評価関数の定義
    criterion = nn.CrossEntropyLoss()

    # model 学習モードに設定
    model.train()
    for name, module in model.named_modules():
        if (name + ".").startswith(frozen_layers):
            module.eval()

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)

        if isinstance(outputs, tuple):
            loss = sum(criterion(logits, labels) for logits in outputs)
        else:
            loss = criterion(outputs, labels)
            
        loss.backward()
        optimizer.step()


def test(model, test_loader, device, exit_id=None):

    # model 評価モードに設定
    model.eval()

    with torch.no_grad():
        correct_preds = 0
        total_preds = 0

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs) if exit_id is None else model.forward_exit(inputs, exit_id)

            correct_preds += outputs.max(1)[1].eq(labels).sum().item()
            total_preds += outputs.size(0)

        acc = 100*correct_preds/total_preds
    return acc