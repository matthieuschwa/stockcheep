import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock(nn.Module):
    """
    Residual block with batch normalization.
    
    Architecture:
        Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> Add -> ReLU
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class ChessNet(nn.Module):
    """
    Chess Neural Network with dual heads (policy and value).
    
    Input: 14×8×8 tensor representing the board state
    Output: 
        - Policy: 4096-dimensional vector (probabilities for all possible moves)
        - Value: Scalar value estimating the position evaluation [-1, 1]
    """
    
    def __init__(self, num_res_blocks: int = 10, num_channels: int = 256):
        super().__init__()
        
        self.num_res_blocks = num_res_blocks
        self.num_channels = num_channels

        self.conv_input = nn.Conv2d(14, num_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])

        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4096)

        self.value_conv = nn.Conv2d(num_channels, 8, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 14, 8, 8)
            
        Returns:
            policy: Tensor of shape (batch_size, 4096) - move probabilities
            value: Tensor of shape (batch_size, 1) - position evaluation
        """
        x = F.relu(self.bn_input(self.conv_input(x)))

        for block in self.res_blocks:
            x = block(x)

        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)

        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value

    def get_device(self) -> torch.device:
        """Get the device the model is on."""
        return next(self.parameters()).device

    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_checkpoint(self, path: str, metadata: dict = None):
        """
        Save model checkpoint with metadata.
        
        Args:
            path: Path to save the checkpoint
            metadata: Additional metadata to save
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'num_res_blocks': self.num_res_blocks,
            'num_channels': self.num_channels,
        }
        if metadata:
            checkpoint.update(metadata)
        torch.save(checkpoint, path)
    
    @classmethod
    def load_checkpoint(cls, path: str, device: torch.device = None):
        """
        Load model from checkpoint.
        
        Args:
            path: Path to the checkpoint
            device: Device to load the model on
            
        Returns:
            Loaded model instance
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            num_res_blocks=checkpoint.get('num_res_blocks', 10),
            num_channels=checkpoint.get('num_channels', 256)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model