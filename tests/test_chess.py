import pytest
import torch
import chess
import numpy as np

from src.ai.model import ChessNet, ResidualBlock
from src.ai.encoder import encode_board, move_to_index, decode_move
from src.game.game_manager import GameManager, GameState


class TestModel:
    """Tests for neural network model."""
    
    def test_residual_block_forward(self):
        """Test residual block forward pass."""
        block = ResidualBlock(64)
        x = torch.randn(1, 64, 8, 8)
        out = block(x)
        assert out.shape == x.shape
    
    def test_chessnet_initialization(self):
        """Test ChessNet initialization."""
        model = ChessNet(num_res_blocks=5, num_channels=128)
        assert model.num_res_blocks == 5
        assert model.num_channels == 128
        assert model.count_parameters() > 0
    
    def test_chessnet_forward(self):
        """Test ChessNet forward pass."""
        model = ChessNet(num_res_blocks=3, num_channels=64)
        x = torch.randn(2, 14, 8, 8)
        policy, value = model(x)
        
        assert policy.shape == (2, 4096)
        assert value.shape == (2, 1)
        assert -1 <= value.min() <= 1
        assert -1 <= value.max() <= 1
    
    def test_model_save_load(self, tmp_path):
        """Test model save and load."""
        model = ChessNet(num_res_blocks=2, num_channels=32)
        
        save_path = tmp_path / "test_model.pth"
        model.save_checkpoint(str(save_path))
        assert save_path.exists()
        
        loaded_model = ChessNet.load_checkpoint(str(save_path))
        assert loaded_model.num_res_blocks == model.num_res_blocks
        assert loaded_model.num_channels == model.num_channels


class TestEncoder:
    """Tests for board encoding."""
    
    def test_encode_initial_position(self):
        """Test encoding of initial board position."""
        board = chess.Board()
        tensor = encode_board(board)
        
        assert tensor.shape == (14, 8, 8)
        assert tensor.dtype == torch.float32
        
        assert tensor[0, 1, :].sum() == 8
        
        assert tensor[6, 6, :].sum() == 8
        
        assert tensor[12].sum() == 64
    
    def test_move_to_index(self):
        """Test move to index conversion."""
        move = chess.Move.from_uci("e2e4")
        idx = move_to_index(move)
        assert 0 <= idx < 4096
        assert idx == move.from_square * 64 + move.to_square
    
    def test_decode_move(self):
        """Test move decoding."""
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        idx = move_to_index(move)
        
        decoded_move = decode_move(board, idx)
        assert decoded_move is not None
        assert decoded_move == move


class TestGameManager:
    """Tests for game management."""
    
    def test_game_initialization(self):
        """Test game initialization."""
        game = GameManager()
        assert game.board.fen() == chess.STARTING_FEN
        assert game.state == GameState.NOT_STARTED
        assert len(game.move_history) == 0
    
    def test_start_game(self):
        """Test starting a game."""
        game = GameManager()
        game.start_game()
        assert game.state == GameState.IN_PROGRESS
        assert game.start_time is not None
    
    def test_make_legal_move(self):
        """Test making a legal move."""
        game = GameManager()
        game.start_game()
        
        success = game.make_move_uci("e2e4")
        assert success
        assert len(game.move_history) == 1
        assert game.board.fen() != chess.STARTING_FEN
    
    def test_make_illegal_move(self):
        """Test making an illegal move."""
        game = GameManager()
        game.start_game()
        
        success = game.make_move_uci("e2e5") 
        assert not success
        assert len(game.move_history) == 0
    
    def test_undo_move(self):
        """Test undoing a move."""
        game = GameManager()
        game.start_game()
        
        game.make_move_uci("e2e4")
        assert len(game.move_history) == 1
        
        success = game.undo_move()
        assert success
        assert len(game.move_history) == 0
        assert game.board.fen() == chess.STARTING_FEN
    
    def test_get_legal_moves(self):
        """Test getting legal moves."""
        game = GameManager()
        game.start_game()
        
        legal_moves = game.get_legal_moves()
        assert len(legal_moves) == 20  # 20 possible opening moves
        assert "e2e4" in legal_moves
    
    def test_checkmate_detection(self):
        """Test checkmate detection."""
        game = GameManager()
        game.start_game()
        
        game.make_move_uci("e2e4")
        game.make_move_uci("e7e5")
        game.make_move_uci("d1h5")
        game.make_move_uci("b8c6")
        game.make_move_uci("f1c4")
        game.make_move_uci("g8f6")
        game.make_move_uci("h5f7")
        
        assert game.state == GameState.CHECKMATE
        assert game.is_game_over()
    
    def test_save_and_load_game(self, tmp_path):
        """Test saving and loading a game."""
        game = GameManager()
        game.start_game()
        game.make_move_uci("e2e4")
        game.make_move_uci("e7e5")
        
        save_path = tmp_path / "test_game.json"
        game.save_game(str(save_path))
        assert save_path.exists()
        
        loaded_game = GameManager.load_game(str(save_path))
        assert loaded_game.board.fen() == game.board.fen()
        assert len(loaded_game.move_history) == len(game.move_history)


class TestIntegration:
    """Integration tests."""
    
    @pytest.mark.slow
    def test_model_inference(self):
        """Test model inference on a position."""
        model = ChessNet(num_res_blocks=2, num_channels=32)
        model.eval()
        
        board = chess.Board()
        board_tensor = encode_board(board).unsqueeze(0)
        
        with torch.no_grad():
            policy, value = model(board_tensor)
        
        assert policy.shape == (1, 4096)
        assert value.shape == (1, 1)
        assert not torch.isnan(policy).any()
        assert not torch.isnan(value).any()
    
    @pytest.mark.slow
    def test_full_game_simulation(self):
        """Test playing a full game."""
        game = GameManager()
        game.start_game()
        
        move_count = 0
        max_moves = 100
        
        while not game.is_game_over() and move_count < max_moves:
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break
            
            import random
            move = random.choice(legal_moves)
            game.make_move_uci(move)
            move_count += 1
        
        assert move_count > 0
        assert len(game.move_history) == move_count


if __name__ == '__main__':
    pytest.main([__file__, '-v'])