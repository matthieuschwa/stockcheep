import chess
import chess.pgn
from enum import Enum
from typing import Optional, List, Dict, Tuple
from datetime import datetime
import json
from pathlib import Path


class GameState(Enum):
    """Enumeration of possible game states."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    CHECKMATE = "checkmate"
    STALEMATE = "stalemate"
    INSUFFICIENT_MATERIAL = "insufficient_material"
    THREEFOLD_REPETITION = "threefold_repetition"
    FIFTY_MOVES = "fifty_moves"
    DRAW_ACCEPTED = "draw_accepted"
    RESIGNED = "resigned"


class GameManager:
    """
    Manages a chess game between two players.
    """
    
    def __init__(self, player_white=None, player_black=None):
        """
        Initialize a new game.
        
        Args:
            player_white: White player instance
            player_black: Black player instance
        """
        self.board = chess.Board()
        self.player_white = player_white
        self.player_black = player_black
        self.state = GameState.NOT_STARTED
        self.move_history: List[chess.Move] = []
        self.time_per_move: List[float] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.result: Optional[str] = None
        self.metadata: Dict = {}

    def start_game(self):
        """Start the game."""
        self.state = GameState.IN_PROGRESS
        self.start_time = datetime.now()
        self.move_history = []
        self.time_per_move = []

    def make_move(self, move: chess.Move, time_taken: float = 0.0) -> bool:
        """
        Make a move on the board.
        
        Args:
            move: The move to make
            time_taken: Time taken for the move in seconds
            
        Returns:
            True if move was successful, False otherwise
        """
        if self.state != GameState.IN_PROGRESS:
            return False
        
        if move not in self.board.legal_moves:
            return False
        
        self.board.push(move)
        self.move_history.append(move)
        self.time_per_move.append(time_taken)
        
        self._update_game_state()
        
        return True

    def make_move_uci(self, uci_move: str) -> bool:
        """
        Make a move using UCI notation.
        
        Args:
            uci_move: Move in UCI format (e.g., "e2e4")
            
        Returns:
            True if move was successful, False otherwise
        """
        try:
            move = chess.Move.from_uci(uci_move)
            return self.make_move(move)
        except:
            return False

    def make_move_san(self, san_move: str) -> bool:
        """
        Make a move using SAN notation.
        
        Args:
            san_move: Move in SAN format (e.g., "Nf3")
            
        Returns:
            True if move was successful, False otherwise
        """
        try:
            move = self.board.parse_san(san_move)
            return self.make_move(move)
        except:
            return False

    def undo_move(self) -> bool:
        """
        Undo the last move.
        
        Returns:
            True if successful, False otherwise
        """
        if len(self.move_history) == 0:
            return False
        
        self.board.pop()
        self.move_history.pop()
        if self.time_per_move:
            self.time_per_move.pop()
        
        self.state = GameState.IN_PROGRESS
        return True

    def get_legal_moves(self) -> List[str]:
        """
        Get all legal moves in UCI format.
        
        Returns:
            List of legal moves in UCI notation
        """
        return [move.uci() for move in self.board.legal_moves]

    def get_legal_moves_san(self) -> List[str]:
        """
        Get all legal moves in SAN format.
        
        Returns:
            List of legal moves in SAN notation
        """
        return [self.board.san(move) for move in self.board.legal_moves]

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.state != GameState.IN_PROGRESS and self.state != GameState.NOT_STARTED

    def _update_game_state(self):
        """Update the game state based on the current position."""
        if self.board.is_checkmate():
            self.state = GameState.CHECKMATE
            self.result = "1-0" if self.board.turn == chess.BLACK else "0-1"
            self.end_time = datetime.now()
        elif self.board.is_stalemate():
            self.state = GameState.STALEMATE
            self.result = "1/2-1/2"
            self.end_time = datetime.now()
        elif self.board.is_insufficient_material():
            self.state = GameState.INSUFFICIENT_MATERIAL
            self.result = "1/2-1/2"
            self.end_time = datetime.now()
        elif self.board.can_claim_threefold_repetition():
            self.state = GameState.THREEFOLD_REPETITION
            self.result = "1/2-1/2"
            self.end_time = datetime.now()
        elif self.board.can_claim_fifty_moves():
            self.state = GameState.FIFTY_MOVES
            self.result = "1/2-1/2"
            self.end_time = datetime.now()

    def resign(self, color: chess.Color):
        """
        Resign the game.
        
        Args:
            color: Color of the player resigning
        """
        self.state = GameState.RESIGNED
        self.result = "0-1" if color == chess.WHITE else "1-0"
        self.end_time = datetime.now()

    def offer_draw(self) -> bool:
        """
        Offer a draw (auto-accepted for now).
        
        Returns:
            True if draw accepted
        """
        self.state = GameState.DRAW_ACCEPTED
        self.result = "1/2-1/2"
        self.end_time = datetime.now()
        return True

    def get_fen(self) -> str:
        """Get the current FEN string."""
        return self.board.fen()

    def get_board_state(self) -> Dict:
        """
        Get complete board state as dictionary.
        
        Returns:
            Dictionary with board state information
        """
        return {
            'fen': self.board.fen(),
            'turn': 'white' if self.board.turn == chess.WHITE else 'black',
            'legal_moves': self.get_legal_moves(),
            'is_check': self.board.is_check(),
            'is_checkmate': self.board.is_checkmate(),
            'is_stalemate': self.board.is_stalemate(),
            'is_game_over': self.is_game_over(),
            'result': self.result,
            'move_count': len(self.move_history),
            'last_move': self.move_history[-1].uci() if self.move_history else None,
            'state': self.state.value
        }

    def get_move_history(self) -> List[Dict]:
        """
        Get the move history with details.
        
        Returns:
            List of move dictionaries
        """
        history = []
        board = chess.Board()
        
        for i, move in enumerate(self.move_history):
            move_dict = {
                'number': (i // 2) + 1,
                'color': 'white' if i % 2 == 0 else 'black',
                'uci': move.uci(),
                'san': board.san(move),
                'time': self.time_per_move[i] if i < len(self.time_per_move) else 0.0
            }
            history.append(move_dict)
            board.push(move)
        
        return history

    def export_pgn(self, filename: Optional[str] = None) -> str:
        """
        Export the game to PGN format.
        
        Args:
            filename: Optional filename to save PGN
            
        Returns:
            PGN string
        """
        game = chess.pgn.Game()
        
        game.headers["Event"] = self.metadata.get("event", "StockCheep")
        game.headers["Date"] = self.start_time.strftime("%Y.%m.%d") if self.start_time else "????.??.??"
        game.headers["White"] = self.metadata.get("white_player", "White")
        game.headers["Black"] = self.metadata.get("black_player", "Black")
        game.headers["Result"] = self.result or "*"
        
        node = game
        for move in self.move_history:
            node = node.add_variation(move)
        
        pgn_string = str(game)
        
        if filename:
            with open(filename, 'w') as f:
                f.write(pgn_string)
        
        return pgn_string

    def save_game(self, filepath: str):
        """
        Save game state to JSON file.
        
        Args:
            filepath: Path to save the game
        """
        game_data = {
            'fen': self.board.fen(),
            'move_history': [move.uci() for move in self.move_history],
            'time_per_move': self.time_per_move,
            'state': self.state.value,
            'result': self.result,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'metadata': self.metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(game_data, f, indent=2)

    @classmethod
    def load_game(cls, filepath: str) -> 'GameManager':
        """
        Load game state from JSON file.
        
        Args:
            filepath: Path to the saved game
            
        Returns:
            GameManager instance
        """
        with open(filepath, 'r') as f:
            game_data = json.load(f)
        
        game = cls()
        game.board = chess.Board(game_data['fen'])
        
        temp_board = chess.Board()
        for uci_move in game_data['move_history']:
            move = chess.Move.from_uci(uci_move)
            game.move_history.append(move)
            temp_board.push(move)
        
        game.time_per_move = game_data.get('time_per_move', [])
        game.state = GameState(game_data['state'])
        game.result = game_data.get('result')
        game.metadata = game_data.get('metadata', {})
        
        if game_data.get('start_time'):
            game.start_time = datetime.fromisoformat(game_data['start_time'])
        if game_data.get('end_time'):
            game.end_time = datetime.fromisoformat(game_data['end_time'])
        
        return game