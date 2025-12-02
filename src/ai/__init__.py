"""
StockCheep Module

This module contains the neural network architecture, MCTS implementation,
and training logic for StockCheep
"""

from .model import ChessNet, ResidualBlock
from .mcts import MCTSNode, mcts_search
from .encoder import encode_board, decode_move, move_to_index
from .trainer import ChessTrainer

__all__ = [
    'ChessNet',
    'ResidualBlock',
    'MCTSNode',
    'mcts_search',
    'encode_board',
    'decode_move',
    'move_to_index',
    'ChessTrainer'
]