import torch
import torch.nn as nn
import numpy as np

class DropBackVGGS(nn.Module):
    def __init__(self, input_channels, output_dim, pruning_threshold, seed):
        super(DropBackVGGS, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, output_dim)
        self.pruning_threshold = pruning_threshold

        self.initial_params = None
        self.initialize_params(seed)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = torch.relu(self.conv2(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = torch.relu(self.conv3(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = x.view(-1, 256 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def initialize_params(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        
        for conv_layer in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_normal_(conv_layer.weight, mode='fan_out', nonlinearity='relu')
            if conv_layer.bias is not None:
                nn.init.zeros_(conv_layer.bias)
        
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.fc2.bias)

        self.initial_params = torch.cat([param.view(-1) for param in self.parameters() if param.requires_grad])

    def freeze_params(self):
        # get current params
        all_params = torch.cat([param.view(-1) for param in self.parameters() if param.requires_grad])
        cumulative_updates = torch.zeros_like(all_params)
        idx = 0
        for param in self.parameters():
            if param.requires_grad:
                param_grad = param.grad.view(-1)
                cumulative_updates[idx:idx+len(param_grad)] += torch.abs(param_grad)
                idx += len(param_grad)

        num_params = sum(param.numel() for param in self.parameters() if param.requires_grad)
        num_freeze = int(self.pruning_threshold * num_params)
        print(f"======== Pruned weights: {num_freeze} out of {num_params} ========")
        self.freeze_indices = np.argsort(cumulative_updates.cpu().numpy())[:num_freeze]

    def reset_frozen(self, device):
        idx = 0
        for param in self.parameters():
            if param.requires_grad:
                param_size = param.numel()
                for freeze_idx in self.freeze_indices:
                    if idx <= freeze_idx < idx + param_size:
                        freeze_idx -= idx
                        param.data.view(-1)[freeze_idx] = self.initial_params[idx + freeze_idx].to(device)
                idx += param_size