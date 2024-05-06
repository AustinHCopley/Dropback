import torch
import torch.nn as nn
import numpy as np

class DropBackMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, pruning_threshold, seed):
        super(DropBackMLP, self).__init__()
        self.pruning_threshold = pruning_threshold
        self.seed = seed

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.initial_params = None
        self.initialize_params()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def initialize_params(self):
        torch.manual_seed(self.seed)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.initialized = True
        # save initialized parameters as one contiguous array
        self.initial_params = torch.cat([param.view(-1) for param in self.parameters() if param.requires_grad])

    def freeze_params(self): # called when freeze epoch ends
        all_params = torch.cat([param.view(-1) for param in self.parameters() if param.requires_grad])
        cumulative_updates = torch.zeros_like(all_params)
        idx = 0
        for param in self.parameters(): # get cumulative gradient updates for params
            if param.requires_grad:
                param_grad = param.grad.view(-1)
                cumulative_updates[idx:idx+len(param_grad)] += torch.abs(param_grad)
                idx += len(param_grad)

        num_params = sum(param.numel() for param in self.parameters() if param.requires_grad)
        num_freeze = int(self.pruning_threshold * num_params)
        print(f"======== Pruned weights: {num_freeze} out of {num_params} ========")
        # save indices of least updated params to restore from initialized vals
        self.freeze_indices = np.argsort(cumulative_updates.cpu().numpy())[:num_freeze]

    def reset_frozen(self, device): # restore frozen params every epoch after freezing
        idx = 0
        for param in self.parameters():
            if param.requires_grad:
                param_size = param.numel()
                for freeze_idx in self.freeze_indices:
                    if idx <= freeze_idx < idx + param_size:
                        freeze_idx -= idx
                        param.data.view(-1)[freeze_idx] = self.initial_params[idx + freeze_idx].to(device)
                idx += param_size
