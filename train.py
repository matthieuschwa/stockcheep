import argparse
import yaml
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.ai.model import ChessNet
from src.ai.trainer import ChessTrainer
from loguru import logger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train StockCheep')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--iterations', type=int, default=None,
                        help='Number of training iterations (overrides config)')
    parser.add_argument('--games', type=int, default=None,
                        help='Games per iteration (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    training_config = config['training']
    
    num_iterations = args.iterations or training_config['num_iterations']
    games_per_iteration = args.games or training_config['games_per_iteration']
    epochs_per_iteration = training_config['epochs_per_iteration']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        model = ChessNet.load_checkpoint(args.resume, device)
    else:
        logger.info("Creating new model")
        model = ChessNet(
            num_res_blocks=config['model']['architecture']['num_res_blocks'],
            num_channels=config['model']['architecture']['num_channels']
        ).to(device)
    
    logger.info(f"Model has {model.count_parameters():,} parameters")
    
    trainer = ChessTrainer(model, training_config, device)
    
    save_dir = Path(config['paths']['checkpoints'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting training...")
    logger.info(f"Iterations: {num_iterations}")
    logger.info(f"Games per iteration: {games_per_iteration}")
    logger.info(f"Epochs per iteration: {epochs_per_iteration}")
    
    trainer.run_training(
        num_iterations=num_iterations,
        games_per_iteration=games_per_iteration,
        epochs_per_iteration=epochs_per_iteration,
        save_dir=str(save_dir)
    )
    
    logger.info("Training completed!")


if __name__ == '__main__':
    main()