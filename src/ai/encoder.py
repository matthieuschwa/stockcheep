import chess
import torch
import numpy as np
from typing import Tuple


def encode_board(board: chess.Board) -> torch.Tensor:
    """
    Encode a chess board into a 14 x 8 x 8 tensor.
    
    14 planes:
        - Planes 0-5: White pieces (Pawn, Knight, Bishop, Rook, Queen, King)
        - Planes 6-11: Black pieces (Pawn, Knight, Bishop, Rook, Queen, King)
        - Plane 12: Current player (1 if white to move, 0 if black)
        - Plane 13: Castling rights
    
    Args:
        board: python-chess Board object
        
    Returns:
        Tensor of shape (14, 8, 8)
    """
    tensor = np.zeros((14, 8, 8), dtype=np.float32)

    piece_to_plane = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank, file = divmod(square, 8)
            plane = piece_to_plane[piece.piece_type]
            
            if piece.color == chess.WHITE:
                tensor[plane, rank, file] = 1
            else:
                tensor[plane + 6, rank, file] = 1

    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1

    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, 0, 6:8] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[13, 0, 0:3] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[13, 7, 6:8] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[13, 7, 0:3] = 1

    return torch.FloatTensor(tensor)


def move_to_index(move: chess.Move) -> int:
    """
    Convert a chess move to an index in the policy vector.
    
    Simplified encoding: from_square * 64 + to_square
    This gives us 64 * 64 = 4096 possible indices.
    
    Args:
        move: chess.Move object
        
    Returns:
        Index in range [0, 4096)
    """
    return move.from_square * 64 + move.to_square


def index_to_move(index: int) -> Tuple[int, int]:
    """
    Convert an index back to from_square and to_square.
    
    Args:
        index: Index in the policy vector
        
    Returns:
        Tuple of (from_square, to_square)
    """
    from_square = index // 64
    to_square = index % 64
    return from_square, to_square


def decode_move(board: chess.Board, index: int) -> chess.Move:
    """
    Convert an index to a valid chess.Move object.
    
    Args:
        board: Current board state
        index: Index in the policy vector
        
    Returns:
        chess.Move object if valid, None otherwise
    """
    from_square, to_square = index_to_move(index)
    
    try:
        move = chess.Move(from_square, to_square)
        
        if board.piece_at(from_square) and board.piece_at(from_square).piece_type == chess.PAWN:
            to_rank = chess.square_rank(to_square)
            if to_rank == 0 or to_rank == 7:
                move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
        
        if move in board.legal_moves:
            return move
    except:
        pass
    
    return None


def get_legal_moves_mask(board: chess.Board) -> np.ndarray:
    """
    Create a mask for legal moves in the current position.
    
    Args:
        board: Current board state
        
    Returns:
        Binary mask of shape (4096,) where 1 indicates a legal move
    """
    mask = np.zeros(4096, dtype=np.float32)
    
    for move in board.legal_moves:
        idx = move_to_index(move)
        mask[idx] = 1
    
    return mask


def mirror_board(board: chess.Board) -> chess.Board:
    """
    Mirror the board (for data augmentation).
    
    Args:
        board: Board to mirror
        
    Returns:
        Mirrored board
    """
    mirrored = chess.Board(None)
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            mirrored_file = 7 - file
            mirrored_square = chess.square(mirrored_file, rank)
            mirrored.set_piece_at(mirrored_square, piece)
    
    mirrored.turn = board.turn
    mirrored.castling_rights = board.castling_rights
    mirrored.ep_square = board.ep_square
    
    return mirrored