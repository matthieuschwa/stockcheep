import argparse
import yaml
import torch
import chess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.ai.model import ChessNet
from src.ai.mcts import select_move
from src.game.game_manager import GameManager
from src.game.player import AIPlayer, HumanPlayer


def display_board(board: chess.Board):
    """Display the chess board in the terminal."""
    print("\n" + "="*40)
    print(board)
    print("="*40)
    print(f"FEN: {board.fen()}")
    print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
    if board.is_check():
        print("CHECK!")
    print()


def get_human_move(board: chess.Board) -> chess.Move:
    """Get a move from the human player."""
    while True:
        try:
            move_str = input("Enter your move (e.g., 'e2e4' or 'Nf3'): ").strip()
            
            if move_str.lower() in ['quit', 'exit', 'q']:
                sys.exit(0)
            
            # Try UCI format first
            try:
                move = chess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    return move
            except:
                pass
            
            # Try SAN format
            try:
                move = board.parse_san(move_str)
                return move
            except:
                pass
            
            print("Invalid move! Try again.")
            print(f"Legal moves: {', '.join([board.san(m) for m in list(board.legal_moves)[:10]])}...")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description='Play Chess against AI')
    parser.add_argument('--model', type=str, default='data/models/chess_model_final.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--color', type=str, default='white', choices=['white', 'black'],
                        help='Your color')
    parser.add_argument('--difficulty', type=str, default='medium',
                        choices=['easy', 'medium', 'hard'],
                        help='AI difficulty')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train a model first using train.py")
        sys.exit(1)
    
    print(f"Loading model from {model_path}...")
    model = ChessNet.load_checkpoint(str(model_path), device)
    print("Model loaded successfully!")
    
    player_color = chess.WHITE if args.color == 'white' else chess.BLACK
    ai_color = chess.BLACK if player_color == chess.WHITE else chess.WHITE
    
    difficulty_settings = config['game']['difficulty_levels'][args.difficulty]
    
    human_player = HumanPlayer("You", player_color)
    ai_player = AIPlayer(
        model,
        f"CheepStock ({args.difficulty.capitalize()})",
        ai_color,
        num_simulations=difficulty_settings['num_simulations'],
        temperature=difficulty_settings['temperature']
    )
    
    print("\n" + "="*40)
    print("CheepStock - COMMAND LINE EDITION  ♟️")
    print("="*40)
    print(f"You are playing as: {args.color.capitalize()}")
    print(f"AI difficulty: {args.difficulty.capitalize()}")
    print(f"AI simulations: {difficulty_settings['num_simulations']}")
    print("\nCommands: 'quit' or 'q' to exit")
    print("Move format: 'e2e4' (UCI) or 'Nf3' (SAN)")
    print("="*40)
    
    if player_color == chess.WHITE:
        game = GameManager(human_player, ai_player)
    else:
        game = GameManager(ai_player, human_player)
    
    game.start_game()
    
    while not game.is_game_over():
        display_board(game.board)
        
        current_player = game.player_white if game.board.turn == chess.WHITE else game.player_black
        
        if isinstance(current_player, HumanPlayer):
            move = get_human_move(game.board)
            game.make_move(move)
        else:
            print("AI is thinking...")
            move = ai_player.get_move(game.board)
            game.make_move(move, ai_player.time_taken[-1])
            print(f"AI plays: {game.board.san(move)} ({move.uci()})")
            print(f"Time taken: {ai_player.time_taken[-1]:.2f}s")
    
    display_board(game.board)
    print("\n" + "="*40)
    print("GAME OVER!")
    print("="*40)
    
    if game.board.is_checkmate():
        winner = "White" if game.board.turn == chess.BLACK else "Black"
        print(f"Checkmate! {winner} wins!")
    elif game.board.is_stalemate():
        print("Stalemate!")
    elif game.board.is_insufficient_material():
        print("Draw by insufficient material")
    elif game.board.can_claim_threefold_repetition():
        print("Draw by threefold repetition")
    elif game.board.can_claim_fifty_moves():
        print("Draw by fifty-move rule")
    
    print(f"Result: {game.result}")
    print("="*40)
    
    save_game = input("\nSave game? (y/n): ").strip().lower()
    if save_game == 'y':
        filename = f"data/games/game_{game.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        game.save_game(filename)
        print(f"Game saved to {filename}")


if __name__ == '__main__':
    main()