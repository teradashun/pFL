import torch
import torch.nn as nn
import torch.nn.functional as F


class CifarBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        if stride not in (1, 2):
            raise ValueError("strideは1または2を指定してください")
        if out_channels < in_channels:
            raise ValueError("出力チャネル数を減らす構造には対応していません")

        self.stride = stride
        self.channel_padding = out_channels - in_channels

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Option A：空間方向の間引き＋チャネル方向のゼロ埋め
        shortcut = x[:, :, ::self.stride, ::self.stride]

        if self.channel_padding:
            left = self.channel_padding // 2
            right = self.channel_padding - left
            shortcut = F.pad(
                shortcut, (0, 0, 0, 0, left, right)
            )

        return F.relu(out + shortcut)


class EENet20(nn.Module):
    exit_ids = (10, 16, 20)

    def __init__(
        self,
        num_classes=10,
        channels=(16, 32, 64),
    ):
        super().__init__()
        c1, c2, c3 = channels
        self.in_channels = c1

        self.conv1 = nn.Conv2d(
            3, c1,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(c1)

        self.layer1 = self._make_layer(c1, stride=1)
        self.layer2 = self._make_layer(c2, stride=2)
        self.layer3 = self._make_layer(c3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.exit10 = nn.Linear(c2, num_classes)
        self.exit16 = nn.Linear(c3, num_classes)
        self.fc = nn.Linear(c3, num_classes)

    def _make_layer(self, out_channels, stride):
        blocks = []

        for block_stride in (stride, 1, 1):
            blocks.append(
                CifarBasicBlock(
                    self.in_channels,
                    out_channels,
                    stride=block_stride,
                )
            )
            self.in_channels = out_channels

        return nn.Sequential(*blocks)

    def _classify(self, x, head):
        return head(torch.flatten(self.avgpool(x), 1))

    def _shared_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        return self.layer2[0](x)

    def _to_exit16(self, x):
        x = self.layer2[1:](x)
        return self.layer3[0](x)

    def forward(self, x):
        x = self._shared_features(x)
        out10 = self._classify(x, self.exit10)

        x = self._to_exit16(x)
        out16 = self._classify(x, self.exit16)

        x = self.layer3[1:](x)
        out20 = self._classify(x, self.fc)

        return out10, out16, out20

    def forward_exit(self, x, exit_id):
        if exit_id not in self.exit_ids:
            raise ValueError(f"未対応のExit: {exit_id}")

        x = self._shared_features(x)

        if exit_id == 10:
            return self._classify(x, self.exit10)

        x = self._to_exit16(x)

        if exit_id == 16:
            return self._classify(x, self.exit16)

        x = self.layer3[1:](x)
        return self._classify(x, self.fc)

    @torch.no_grad()
    def forward_early(self, x, thresholds):
        if len(thresholds) != 2:
            raise ValueError("閾値は2個指定してください")

        result = x.new_empty((x.size(0), self.fc.out_features))
        remaining = torch.arange(x.size(0), device=x.device)

        x = self._shared_features(x)

        for index, head in enumerate((self.exit10, self.exit16)):
            if index == 1:
                x = self._to_exit16(x)

            logits = self._classify(x, head)
            log_probs = F.log_softmax(logits, dim=1)
            entropy = -(log_probs.exp() * log_probs).sum(dim=1)
            done = entropy <= thresholds[index]

            result[remaining[done]] = logits[done]
            remaining = remaining[~done]
            x = x[~done]

            if remaining.numel() == 0:
                return result

        x = self.layer3[1:](x)
        result[remaining] = self._classify(x, self.fc)
        return result


def eenet20_half(num_classes=10):
    return EENet20(
        num_classes=num_classes,
        channels=(16, 16, 32),
    )