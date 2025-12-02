import chess
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .encoder import encode_board, move_to_index, get_legal_moves_mask


@dataclass
class MCTSConfig:
    """Configuration for MCTS."""
    num_simulations: int = 200
    c_puct: float = 1.5
    temperature: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25


class MCTSNode:
    """
    Node in the Monte Carlo Tree Search.
    
    Attributes:
        board: Current board state
        parent: Parent node
        move: Move that led to this node
        children: Dictionary of child nodes
        visits: Number of visits to this node
        value_sum: Sum of values backed up to this node
        prior: Prior probability from the neural network
    """
    
    def __init__(
        self,
        board: chess.Board,
        parent: Optional['MCTSNode'] = None,
        move: Optional[chess.Move] = None,
        prior: float = 0.0
    ):
        self.board = board
        self.parent = parent
        self.move = move
        self.children: Dict[chess.Move, MCTSNode] = {}
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior

    def is_leaf(self) -> bool:
        """Check if this node is a leaf node."""
        return len(self.children) == 0

    def value(self) -> float:
        """Get the average value of this node."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def ucb_score(self, parent_visits: int, c_puct: float = 1.5) -> float:
        """
        Calculate the Upper Confidence Bound score.
        
        UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        
        Args:
            parent_visits: Number of visits to parent node
            c_puct: Exploration constant
            
        Returns:
            UCB score
        """
        q_value = self.value()
        
        u_value = c_puct * self.prior * np.sqrt(parent_visits) / (1 + self.visits)
        
        return q_value + u_value

    def select_child(self, c_puct: float = 1.5) -> 'MCTSNode':
        """
        Select the best child node based on UCB scores.
        
        Args:
            c_puct: Exploration constant
            
        Returns:
            Selected child node
        """
        best_score = -float('inf')
        best_child = None

        for child in self.children.values():
            score = child.ucb_score(self.visits, c_puct)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def expand(self, model: torch.nn.Module, device: torch.device):
        """
        Expand this node by creating child nodes for all legal moves.
        
        Args:
            model: Neural network model for prior evaluation
            device: Device to run the model on
        """
        if self.board.is_game_over():
            return

        board_tensor = encode_board(self.board).unsqueeze(0).to(device)
        
        with torch.no_grad():
            policy, _ = model(board_tensor)
            policy = F.softmax(policy, dim=1).cpu().numpy()[0]

        legal_mask = get_legal_moves_mask(self.board)
        
        policy = policy * legal_mask
        
        if policy.sum() > 0:
            policy = policy / policy.sum()
        else:
            policy = legal_mask / legal_mask.sum()

        for move in self.board.legal_moves:
            move_idx = move_to_index(move)
            prior = policy[move_idx]
            
            child_board = self.board.copy()
            child_board.push(move)
            
            child = MCTSNode(child_board, parent=self, move=move, prior=prior)
            self.children[move] = child

    def add_exploration_noise(self, dirichlet_alpha: float = 0.3, epsilon: float = 0.25):
        """
        Add Dirichlet noise to the prior probabilities for exploration.
        
        Args:
            dirichlet_alpha: Alpha parameter for Dirichlet distribution
            epsilon: Mixing parameter for noise
        """
        if not self.children:
            return
        
        noise = np.random.dirichlet([dirichlet_alpha] * len(self.children))
        
        for i, child in enumerate(self.children.values()):
            child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]


def mcts_search(
    board: chess.Board,
    model: torch.nn.Module,
    config: MCTSConfig = None
) -> Tuple[List[chess.Move], np.ndarray]:
    """
    Perform Monte Carlo Tree Search to find the best move.
    
    Args:
        board: Current board state
        model: Neural network model
        config: MCTS configuration
        
    Returns:
        Tuple of (moves, visit_probabilities)
    """
    if config is None:
        config = MCTSConfig()
    
    device = next(model.parameters()).device
    root = MCTSNode(board.copy())
    
    root.expand(model, device)
    
    if config.dirichlet_epsilon > 0:
        root.add_exploration_noise(config.dirichlet_alpha, config.dirichlet_epsilon)

    for _ in range(config.num_simulations):
        node = root
        search_board = board.copy()

        # traverse tree to leaf node
        while not node.is_leaf() and not search_board.is_game_over():
            node = node.select_child(config.c_puct)
            search_board.push(node.move)

        if not search_board.is_game_over():
           
            node.expand(model, device)
            
            board_tensor = encode_board(search_board).unsqueeze(0).to(device)
            with torch.no_grad():
                _, value = model(board_tensor)
            value = value.item()
        else:
            result = search_board.result()
            if result == "1-0":
                value = 1.0 if board.turn == chess.WHITE else -1.0
            elif result == "0-1":
                value = -1.0 if board.turn == chess.WHITE else 1.0
            else:
                value = 0.0

        while node is not None:
            node.visits += 1
            node.value_sum += value
            value = -value 
            node = node.parent

    moves = []
    visit_counts = []
    
    for move, child in root.children.items():
        moves.append(move)
        visit_counts.append(child.visits)

    visit_counts = np.array(visit_counts, dtype=np.float32)
    
    if config.temperature == 0:
        # deterministicc: choose most visited
        probabilities = np.zeros_like(visit_counts)
        probabilities[np.argmax(visit_counts)] = 1.0
    else:

        visit_counts = visit_counts ** (1.0 / config.temperature)
        probabilities = visit_counts / visit_counts.sum()

    return moves, probabilities


def select_move(
    board: chess.Board,
    model: torch.nn.Module,
    num_simulations: int = 200,
    temperature: float = 0.0
) -> chess.Move:
    """
    Select a move using MCTS.
    
    Args:
        board: Current board state
        model: Neural network model
        num_simulations: Number of MCTS simulations
        temperature: Temperature for move selection (0 = deterministic)
        
    Returns:
        Selected move
    """
    config = MCTSConfig(
        num_simulations=num_simulations,
        temperature=temperature,
        dirichlet_epsilon=0.0 
    )
    
    moves, probabilities = mcts_search(board, model, config)
    
    if temperature == 0:
        # deterministic
        move_idx = np.argmax(probabilities)
    else:
        # stoch
        move_idx = np.random.choice(len(moves), p=probabilities)
    
    return moves[move_idx]