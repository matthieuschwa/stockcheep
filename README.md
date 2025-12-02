# stockcheep, cheap edition of stockfish



Fast reminder for non chess player, stockfish is an RL model that combines a deep neural network with Monte Carlo Tree Search (MCTS) to play chess at different skill levels. My Stockcheep is built with PyTorch and FastAPI.

## Quick Start

```bash
docker compose up
```

Then open http://localhost:8000 in your browser.

## How to Play

1. **Choose Your Color**: Select White or Black
2. **Select Difficulty**: 
   - **Easy**: Good for beginners (50 MCTS simulations)
   - **Medium**: Balanced gameplay (200 simulations)
   - **Hard**: Challenging opponent (800 simulations)
3. **Make Moves**: Click on a piece to see legal moves, click destination to move
4. **Enjoy**: The cheep will think and respond with its move

## How It Works

### Architecture Overview

```
┌─────────────────┐
│  Chess Board    │
│   (FEN input)   │
└────────┬────────┘
         │
    ┌────▼────┐
    │ Encoder │  ← Converts board to 119-plane representation
    └────┬────┘
         │
    ┌────▼────────────┐
    │ Neural Network  │  ← 8 ResNet blocks, 256 channels
    │  - Policy Head  │  ← Predicts move probabilities
    │  - Value Head   │  ← Evaluates position strength
    └────┬────────────┘
         │
    ┌────▼────┐
    │  MCTS   │  ← Explores move tree
    └────┬────┘
         │
    ┌────▼────┐
    │  Move   │
    └─────────┘
```

### Key Components

1. **Encoder** (`src/ai/encoder.py`)
   - Converts chess positions to 119-plane tensor representation
   - Encodes: piece positions, repetitions, castling rights, en passant

2. **Neural Network** (`src/ai/model.py`)
   - ResNet architecture with 8 residual blocks
   - Two heads: Policy (move prediction) + Value (position evaluation)
   - ~2.4M parameters

3. **MCTS** (`src/ai/mcts.py`)
   - Monte Carlo Tree Search for move selection
   - Uses neural network for leaf evaluation
   - Balances exploration vs exploitation

4. **Trainer** (`src/ai/trainer.py`)
   - Self-play training loop
   - Generates training data from AI vs AI games
   - Optimizes both policy and value predictions

## Training Your Own Model

### Quick Training

```bash
# Train with default settings (100 games, 200 simulations)
python train.py

# Custom training
python train.py --games 500 --simulations 400 --blocks 12 --channels 256
```

### Training Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--games` | 100 | Number of self-play games |
| `--simulations` | 200 | MCTS simulations per move |
| `--blocks` | 8 | Number of residual blocks |
| `--channels` | 256 | Number of channels |
| `--batch-size` | 256 | Training batch size |
| `--epochs` | 10 | Training epochs per iteration |

### Training Output

```
Training Progress:
Game 1/100: White wins in 45 moves
Game 2/100: Black wins in 62 moves
...
Epoch 1/10: Loss=2.341 (Policy=1.234, Value=1.107)
...
Model saved to: data/models/chess_model_final.pth
```

## 📊 Difficulty Settings

| Difficulty | Simulations | Temperature | Response Time | Strength |
|------------|-------------|-------------|---------------|----------|
| Easy       | 50          | 1.5         | ~1-2 sec      | ⭐       |
| Medium     | 200         | 1.0         | ~3-5 sec      | ⭐⭐⭐    |
| Hard       | 800         | 0.5         | ~10-15 sec    | ⭐⭐⭐⭐⭐ |

**Note**: The same trained model is used for all difficulties. Higher difficulty = more thinking time (MCTS simulations).

## Resources

- [AlphaZero Paper](https://arxiv.org/abs/1712.01815) - Inspiration for this project
- [python-chess](https://python-chess.readthedocs.io/) - Chess library
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework

## Acknowledgments

- **python-chess** by Niklas Fiekas for the chess engine
- **Chessboard.js** by Chris Oakman for the UI
- **AlphaZero** team at DeepMind for the algorithmic approach
- The chess and AI communities for inspiration

*Happy Chess Playing!* 🎮
