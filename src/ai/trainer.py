import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import chess
from collections import deque
from typing import List, Tuple, Dict
from tqdm import tqdm
from pathlib import Path

from .model import ChessNet
from .mcts import mcts_search, MCTSConfig
from .encoder import encode_board, move_to_index


class ChessTrainer:
    """
    Trainer for the StockCheep using self-play and reinforcement learning.
    """
    
    def __init__(
        self,
        model: ChessNet,
        config: Dict,
        device: torch.device
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Neural network model
            config: Training configuration dictionary
            device: Device to train on
        """
        self.model = model
        self.config = config
        self.device = device
        
        self.batch_size = config.get('batch_size', 256)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.weight_decay = config.get('weight_decay', 1e-4)
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10,
            gamma=0.9
        )
        self.scaler = GradScaler()
        
        buffer_size = config.get('replay_buffer_size', 50000)
        self.replay_buffer = deque(maxlen=buffer_size)
        
        self.mcts_config = MCTSConfig(
            num_simulations=config.get('num_mcts_sims', 50),
            c_puct=config.get('c_puct', 1.5),
            temperature=config.get('temperature', 1.0),
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25
        )

    def play_game(self) -> List[Tuple[torch.Tensor, np.ndarray, float]]:
        """
        Play a complete game in self-play mode.
        
        Returns:
            List of (board_tensor, policy_target, value_target) tuples
        """
        board = chess.Board()
        game_data = []
        move_count = 0
        temperature_threshold = self.config.get('temperature_threshold', 30)

        while not board.is_game_over() and move_count < 500:
            self.mcts_config.temperature = 1.0 if move_count < temperature_threshold else 0.1
            
            moves, probabilities = mcts_search(board, self.model, self.mcts_config)
            
            board_tensor = encode_board(board)
            
            policy_target = np.zeros(4096, dtype=np.float32)
            for move, prob in zip(moves, probabilities):
                policy_target[move_to_index(move)] = prob
            
            game_data.append((board_tensor, policy_target, None))
            
            move_idx = np.random.choice(len(moves), p=probabilities)
            board.push(moves[move_idx])
            move_count += 1

        result = board.result()
        if result == "1-0":
            game_result = 1.0
        elif result == "0-1":
            game_result = -1.0
        else:
            game_result = 0.0

        final_game_data = []
        for i, (board_tensor, policy_target, _) in enumerate(game_data):
            value = game_result if i % 2 == 0 else -game_result
            final_game_data.append((board_tensor, policy_target, value))

        return final_game_data

    def generate_self_play_data(self, num_games: int) -> List:
        """
        Generate training data through self-play.
        
        Args:
            num_games: Number of games to play
            
        Returns:
            List of training examples
        """
        all_data = []
        
        for game_num in tqdm(range(num_games), desc="Self-play games"):
            game_data = self.play_game()
            all_data.extend(game_data)
            
            if (game_num + 1) % 10 == 0:
                print(f"Completed {game_num + 1}/{num_games} games, "
                      f"Generated {len(all_data)} positions")

        return all_data

    def train_on_batch(
        self,
        boards: torch.Tensor,
        policy_targets: torch.Tensor,
        value_targets: torch.Tensor
    ) -> Tuple[float, float, float]:
        """
        Train on a single batch.
        
        Args:
            boards: Batch of board tensors
            policy_targets: Batch of policy targets
            value_targets: Batch of value targets
            
        Returns:
            Tuple of (total_loss, policy_loss, value_loss)
        """
        boards = boards.to(self.device)
        policy_targets = policy_targets.to(self.device)
        value_targets = value_targets.to(self.device)

        self.optimizer.zero_grad()

        with autocast():
            policy_out, value_out = self.model(boards)

            policy_loss = -torch.mean(
                torch.sum(policy_targets * F.log_softmax(policy_out, dim=1), dim=1)
            )

            value_loss = F.mse_loss(value_out, value_targets)

            loss = policy_loss + value_loss

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item(), policy_loss.item(), value_loss.item()

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader for training data
            
        Returns:
            Dictionary of average losses
        """
        self.model.train()
        
        total_loss = 0.0
        policy_loss_total = 0.0
        value_loss_total = 0.0
        num_batches = 0

        for boards, policy_targets, value_targets in dataloader:
            loss, policy_loss, value_loss = self.train_on_batch(
                boards, policy_targets, value_targets
            )
            
            total_loss += loss
            policy_loss_total += policy_loss
            value_loss_total += value_loss
            num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'policy_loss': policy_loss_total / num_batches,
            'value_loss': value_loss_total / num_batches
        }

    def train(self, train_data: List, epochs: int) -> List[Dict[str, float]]:
        """
        Train the model on the provided data.
        
        Args:
            train_data: List of training examples
            epochs: Number of epochs to train
            
        Returns:
            List of loss dictionaries for each epoch
        """
        # Prepare dataset
        boards = torch.stack([x[0] for x in train_data])
        policy_targets = torch.FloatTensor([x[1] for x in train_data])
        value_targets = torch.FloatTensor([[x[2]] for x in train_data])

        dataset = TensorDataset(boards, policy_targets, value_targets)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )

        history = []

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            losses = self.train_epoch(dataloader)
            history.append(losses)
            
            print(f"Loss: {losses['loss']:.4f}, "
                  f"Policy Loss: {losses['policy_loss']:.4f}, "
                  f"Value Loss: {losses['value_loss']:.4f}")
            
            self.scheduler.step()

        return history

    def training_iteration(
        self,
        iteration: int,
        num_games: int,
        epochs: int,
        save_dir: Path
    ):
        """
        Perform one complete training iteration.
        
        Args:
            iteration: Current iteration number
            num_games: Number of self-play games
            epochs: Number of training epochs
            save_dir: Directory to save checkpoints
        """
        print(f"\n{'='*70}")
        print(f"TRAINING ITERATION {iteration}")
        print(f"{'='*70}\n")

        # Phase 1: Self-play
        print("Phase 1: Generating self-play data")
        new_data = self.generate_self_play_data(num_games)
        self.replay_buffer.extend(new_data)
        print(f"Buffer size: {len(self.replay_buffer)} positions")

        # Phase 2: Training
        print("\nPhase 2: Training model")
        history = self.train(list(self.replay_buffer), epochs)

        # Phase 3: Save checkpoint
        print("\nPhase 3: Saving checkpoint")
        checkpoint_path = save_dir / f"checkpoint_iter_{iteration}.pth"
        self.model.save_checkpoint(
            str(checkpoint_path),
            metadata={
                'iteration': iteration,
                'buffer_size': len(self.replay_buffer),
                'history': history
            }
        )
        print(f"Checkpoint saved to {checkpoint_path}")

    def run_training(
        self,
        num_iterations: int,
        games_per_iteration: int,
        epochs_per_iteration: int,
        save_dir: str
    ):
        """
        Run the complete training loop.
        
        Args:
            num_iterations: Number of training iterations
            games_per_iteration: Games to play per iteration
            epochs_per_iteration: Training epochs per iteration
            save_dir: Directory to save checkpoints
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        for iteration in range(1, num_iterations + 1):
            self.training_iteration(
                iteration,
                games_per_iteration,
                epochs_per_iteration,
                save_path
            )

        # Save final model
        final_path = save_path / "chess_model_final.pth"
        self.model.save_checkpoint(str(final_path))
        print(f"\n{'='*70}")
        print(f"Training complete! Final model saved to {final_path}")
        print(f"{'='*70}\n")