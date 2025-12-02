import chess
import torch
from abc import ABC, abstractmethod
from typing import Optional
import time

from ..ai.mcts import select_move
from ..ai.model import ChessNet


class Player(ABC):
    """Abstract base class for a chess player."""
    
    def __init__(self, name: str, color: chess.Color):
        """
        Initialize a player.
        
        Args:
            name: Player name
            color: Player color (chess.WHITE or chess.BLACK)
        """
        self.name = name
        self.color = color
        self.time_taken = []

    @abstractmethod
    def get_move(self, board: chess.Board) -> chess.Move:
        """
        Get the next move from the player.
        
        Args:
            board: Current board state
            
        Returns:
            Selected move
        """
        pass

    def get_average_time(self) -> float:
        """Get average time per move."""
        if not self.time_taken:
            return 0.0
        return sum(self.time_taken) / len(self.time_taken)


class HumanPlayer(Player):
    """
    Human player that receives moves through external input.
    """
    
    def __init__(self, name: str = "Human", color: chess.Color = chess.WHITE):
        super().__init__(name, color)
        self.next_move: Optional[chess.Move] = None

    def get_move(self, board: chess.Board) -> chess.Move:
        """
        Get move from human player.
        This is a placeholder - actual move should be set via set_move().
        
        Args:
            board: Current board state
            
        Returns:
            The move set by set_move()
        """
        if self.next_move is None:
            raise ValueError("No move has been set for human player")
        
        move = self.next_move
        self.next_move = None
        return move

    def set_move(self, move: chess.Move):
        """
        Set the next move for the human player.
        
        Args:
            move: Move to set
        """
        self.next_move = move


class AIPlayer(Player):
    """
    AI player that uses neural network and MCTS.
    """
    
    def __init__(
        self,
        model: ChessNet,
        name: str = "StockCheep",
        color: chess.Color = chess.BLACK,
        num_simulations: int = 200,
        temperature: float = 0.0
    ):
        """
        Initialize AI player.
        
        Args:
            model: Neural network model
            name: Player name
            color: Player color
            num_simulations: Number of MCTS simulations
            temperature: Temperature for move selection
        """
        super().__init__(name, color)
        self.model = model
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.device = next(model.parameters()).device

    def get_move(self, board: chess.Board) -> chess.Move:
        """
        Get move from AI player using MCTS.
        
        Args:
            board: Current board state
            
        Returns:
            Selected move
        """
        start_time = time.time()
        
        move = select_move(
            board,
            self.model,
            num_simulations=self.num_simulations,
            temperature=self.temperature
        )
        
        elapsed = time.time() - start_time
        self.time_taken.append(elapsed)
        
        return move

    def set_difficulty(self, difficulty: str):
        """
        Set AI difficulty level.
        
        Args:
            difficulty: One of 'easy', 'medium', 'hard'
        """
        difficulty_settings = {
            'easy': {'simulations': 50, 'temperature': 1.5},
            'medium': {'simulations': 200, 'temperature': 1.0},
            'hard': {'simulations': 800, 'temperature': 0.5}
        }
        
        if difficulty in difficulty_settings:
            settings = difficulty_settings[difficulty]
            self.num_simulations = settings['simulations']
            self.temperature = settings['temperature']
        else:
            raise ValueError(f"Invalid difficulty: {difficulty}")


class RandomPlayer(Player):
    """
    Player that makes random legal moves (for testing).
    """
    
    def __init__(self, name: str = "Random", color: chess.Color = chess.BLACK):
        super().__init__(name, color)
        import random
        self.random = random

    def get_move(self, board: chess.Board) -> chess.Move:
        """
        Get a random legal move.
        
        Args:
            board: Current board state
            
        Returns:
            Random legal move
        """
        legal_moves = list(board.legal_moves)
        return self.random.choice(legal_moves)