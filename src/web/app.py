from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import torch
import chess
import json
import yaml
from pathlib import Path
from loguru import logger
import sys

from ..ai.model import ChessNet
from ..ai.mcts import select_move
from ..game.game_manager import GameManager, GameState
from ..game.player import AIPlayer, HumanPlayer

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add("logs/chess_ai.log", rotation="100 MB", retention="30 days", level="DEBUG")

app = FastAPI(
    title="StockCheep API",
    description="Play chess against an AI powered by neural networks and MCTS",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_games: Dict[str, GameManager] = {}
ai_model: Optional[ChessNet] = None
config: Dict = {}


class MoveRequest(BaseModel):
    game_id: str
    move: str  # UCI format
    

class NewGameRequest(BaseModel):
    player_color: str = "white"  # white or black
    difficulty: str = "medium"  # easy, medium, hard


class GameResponse(BaseModel):
    game_id: str
    state: Dict


@app.on_event("startup")
async def startup_event():
    """Load model and configuration on startup."""
    global ai_model, config
    
    logger.info("Starting StockCheep server...")
    
    config_path = Path("config/config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded")
    else:
        logger.warning("Configuration file not found, using defaults")
        config = {
            'model': {'path': 'data/models/chess_model_final.pth'},
            'mcts': {'num_simulations': 200}
        }
    
    model_path = Path(config['model']['path'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_path.exists():
        try:
            ai_model = ChessNet.load_checkpoint(str(model_path), device)
            logger.info(f"Model loaded from {model_path} on {device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            ai_model = ChessNet().to(device)
            logger.warning("Created new model")
    else:
        logger.warning(f"Model file not found at {model_path}, creating new model")
        ai_model = ChessNet().to(device)
    
    logger.info("StockCheep server started successfully")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page."""
    html_file = Path("src/web/static/index.html")
    if html_file.exists():
        return FileResponse(html_file)
    return HTMLResponse(content=get_default_html(), status_code=200)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": ai_model is not None,
        "active_games": len(active_games),
        "device": str(next(ai_model.parameters()).device) if ai_model else "none"
    }


@app.post("/api/game/new", response_model=GameResponse)
async def new_game(request: NewGameRequest):
    """
    Create a new game.
    
    Args:
        request: New game request with player preferences
        
    Returns:
        Game response with game ID and initial state
    """
    if ai_model is None:
        raise HTTPException(status_code=503, detail="AI model not loaded")
    
    import uuid
    game_id = str(uuid.uuid4())
    
    player_color = chess.WHITE if request.player_color.lower() == "white" else chess.BLACK
    ai_color = chess.BLACK if player_color == chess.WHITE else chess.WHITE
    
    human_player = HumanPlayer("You", player_color)
    
    difficulty_settings = {
        'easy': {'simulations': 50, 'temperature': 1.5},
        'medium': {'simulations': 200, 'temperature': 1.0},
        'hard': {'simulations': 800, 'temperature': 0.5}
    }
    
    settings = difficulty_settings.get(request.difficulty, difficulty_settings['medium'])
    ai_player = AIPlayer(
        ai_model,
        name=f"StockCheep ({request.difficulty.capitalize()})",
        color=ai_color,
        num_simulations=settings['simulations'],
        temperature=settings['temperature']
    )
    
    if player_color == chess.WHITE:
        game = GameManager(human_player, ai_player)
    else:
        game = GameManager(ai_player, human_player)
    
    game.start_game()
    game.metadata = {
        "player_color": request.player_color,
        "difficulty": request.difficulty,
        "white_player": "You" if player_color == chess.WHITE else ai_player.name,
        "black_player": "You" if player_color == chess.BLACK else ai_player.name
    }
    
    active_games[game_id] = game
    
    if ai_color == chess.WHITE:
        ai_move = ai_player.get_move(game.board)
        game.make_move(ai_move, ai_player.time_taken[-1])
    
    logger.info(f"New game created: {game_id} (Player: {request.player_color}, Difficulty: {request.difficulty})")
    
    return GameResponse(
        game_id=game_id,
        state=game.get_board_state()
    )


@app.post("/api/game/{game_id}/move")
async def make_move(game_id: str, request: MoveRequest):
    """
    Make a move in the game.
    
    Args:
        game_id: Game identifier
        request: Move request
        
    Returns:
        Updated game state
    """
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = active_games[game_id]
    
    if game.is_game_over():
        raise HTTPException(status_code=400, detail="Game is already over")
    
    try:
        success = game.make_move_uci(request.move)
        if not success:
            raise HTTPException(status_code=400, detail="Invalid move")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid move: {str(e)}")
    
    logger.info(f"Game {game_id}: Player move {request.move}")
    
    if game.is_game_over():
        return {"state": game.get_board_state(), "ai_move": None}
    
    ai_player = game.player_black if game.board.turn == chess.BLACK else game.player_white
    if isinstance(ai_player, AIPlayer):
        ai_move = ai_player.get_move(game.board)
        game.make_move(ai_move, ai_player.time_taken[-1])
        logger.info(f"Game {game_id}: AI move {ai_move.uci()}")
        
        return {
            "state": game.get_board_state(),
            "ai_move": ai_move.uci()
        }
    
    return {"state": game.get_board_state(), "ai_move": None}


@app.get("/api/game/{game_id}/state")
async def get_game_state(game_id: str):
    """
    Get current game state.
    
    Args:
        game_id: Game identifier
        
    Returns:
        Current game state
    """
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = active_games[game_id]
    return {"state": game.get_board_state()}


@app.get("/api/game/{game_id}/history")
async def get_game_history(game_id: str):
    """
    Get game move history.
    
    Args:
        game_id: Game identifier
        
    Returns:
        Move history
    """
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = active_games[game_id]
    return {"history": game.get_move_history()}


@app.post("/api/game/{game_id}/resign")
async def resign_game(game_id: str):
    """
    Resign the game.
    
    Args:
        game_id: Game identifier
        
    Returns:
        Updated game state
    """
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = active_games[game_id]
    game.resign(game.board.turn)
    
    logger.info(f"Game {game_id}: Player resigned")
    
    return {"state": game.get_board_state()}


@app.delete("/api/game/{game_id}")
async def delete_game(game_id: str):
    """
    Delete a game.
    
    Args:
        game_id: Game identifier
        
    Returns:
        Success message
    """
    if game_id in active_games:
        del active_games[game_id]
        logger.info(f"Game {game_id}: Deleted")
        return {"message": "Game deleted"}
    
    raise HTTPException(status_code=404, detail="Game not found")


@app.get("/api/games")
async def list_games():
    """
    List all active games.
    
    Returns:
        List of active game IDs
    """
    return {
        "games": [
            {
                "game_id": game_id,
                "state": game.state.value,
                "move_count": len(game.move_history)
            }
            for game_id, game in active_games.items()
        ]
    }


def get_default_html() -> str:
    """Return default HTML if static file not found."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>StockCheep</title>
    </head>
    <body>
        <h1>StockCheep Server</h1>
        <p>The server is running. Please access the web interface at the proper URL.</p>
        <p>API Documentation: <a href="/docs">/docs</a></p>
    </body>
    </html>
    """


# mount static files (if they exist)
static_dir = Path("src/web/static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")