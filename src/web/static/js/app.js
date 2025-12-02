class StockCheep {
    constructor() {
        this.gameId = null;
        this.playerColor = 'white';
        this.difficulty = 'medium';
        this.board = null;
        this.game = new Chess();
        this.isPlayerTurn = true;
        
        this.initializeElements();
        this.attachEventListeners();
    }

    initializeElements() {
        this.setupPanel = document.getElementById('setup-panel');
        this.gamePanel = document.getElementById('game-panel');
        
        this.startGameBtn = document.getElementById('start-game-btn');
        this.resignBtn = document.getElementById('resign-btn');
        this.newGameBtn = document.getElementById('new-game-btn');
        this.modalNewGameBtn = document.getElementById('modal-new-game-btn');
        
        this.statusText = document.getElementById('status-text');
        this.turnText = document.getElementById('turn-text');
        this.moveList = document.getElementById('move-list');
        
        this.loadingOverlay = document.getElementById('loading-overlay');
        this.gameOverModal = document.getElementById('game-over-modal');
        this.gameOverMessage = document.getElementById('game-over-message');
    }

    attachEventListeners() {
        this.startGameBtn.addEventListener('click', () => this.startNewGame());
        this.resignBtn.addEventListener('click', () => this.resignGame());
        this.newGameBtn.addEventListener('click', () => this.resetToSetup());
        this.modalNewGameBtn.addEventListener('click', () => this.resetToSetup());
    }

    async startNewGame() {
        this.playerColor = document.querySelector('input[name="color"]:checked').value;
        this.difficulty = document.querySelector('input[name="difficulty"]:checked').value;

        this.showLoading(true);

        try {
            const response = await fetch('/api/game/new', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    player_color: this.playerColor,
                    difficulty: this.difficulty
                })
            });

            if (!response.ok) {
                throw new Error('Failed to create game');
            }

            const data = await response.json();
            this.gameId = data.game_id;

            this.initializeChessboard();
            
            this.setupPanel.style.display = 'none';
            this.gamePanel.style.display = 'flex';
            
            this.updateGameState(data.state);

            console.log('Game started:', this.gameId);

        } catch (error) {
            console.error('Error starting game:', error);
            alert('Failed to start game. Please try again.');
        } finally {
            this.showLoading(false);
        }
    }

    initializeChessboard() {
        const config = {
            draggable: true,
            position: 'start',
            onDragStart: this.onDragStart.bind(this),
            onDrop: this.onDrop.bind(this),
            onSnapEnd: this.onSnapEnd.bind(this),
            pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png'
        };

        if (this.playerColor === 'black') {
            config.orientation = 'black';
        }

        this.board = Chessboard('board', config);
        this.game = new Chess();
    }

    onDragStart(source, piece, position, orientation) {
        if (this.game.game_over()) return false;

        if (!this.isPlayerTurn) return false;

        if ((this.playerColor === 'white' && piece.search(/^b/) !== -1) ||
            (this.playerColor === 'black' && piece.search(/^w/) !== -1)) {
            return false;
        }

        return true;
    }

    async onDrop(source, target) {
        const move = this.game.move({
            from: source,
            to: target,
            promotion: 'q'
        });

        if (move === null) return 'snapback';

        await this.sendMove(move.from + move.to);
    }

    onSnapEnd() {
        this.board.position(this.game.fen());
    }

    async sendMove(moveUci) {
        this.showLoading(true);
        this.isPlayerTurn = false;

        try {
            const response = await fetch(`/api/game/${this.gameId}/move`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    game_id: this.gameId,
                    move: moveUci
                })
            });

            if (!response.ok) {
                throw new Error('Invalid move');
            }

            const data = await response.json();
            
            this.updateGameState(data.state);

            if (data.ai_move) {
                await this.delay(300); // small delay for better UX
                this.game.move({
                    from: data.ai_move.substring(0, 2),
                    to: data.ai_move.substring(2, 4),
                    promotion: data.ai_move.length > 4 ? data.ai_move[4] : undefined
                });
                this.board.position(this.game.fen());
                this.updateGameState(data.state);
            }

            this.isPlayerTurn = true;

        } catch (error) {
            console.error('Error making move:', error);
            alert('Invalid move. Please try again.');
            this.game.undo();
            this.board.position(this.game.fen());
            this.isPlayerTurn = true;
        } finally {
            this.showLoading(false);
        }
    }

    updateGameState(state) {
        if (state.fen !== this.game.fen()) {
            this.game.load(state.fen);
            this.board.position(state.fen);
        }

        this.turnText.textContent = state.turn.charAt(0).toUpperCase() + state.turn.slice(1);

        if (state.is_checkmate) {
            this.statusText.textContent = 'Checkmate!';
            this.showGameOver(`Checkmate! ${state.result === '1-0' ? 'White' : 'Black'} wins!`);
        } else if (state.is_stalemate) {
            this.statusText.textContent = 'Stalemate';
            this.showGameOver('Game drawn by stalemate');
        } else if (state.is_check) {
            this.statusText.textContent = 'Check!';
        } else if (state.is_game_over) {
            this.statusText.textContent = 'Game Over';
            this.showGameOver(`Game Over: ${state.result}`);
        } else {
            this.statusText.textContent = this.isPlayerTurn ? 'Your turn' : 'AI is thinking...';
        }

        this.updateMoveHistory();
    }

    async updateMoveHistory() {
        try {
            const response = await fetch(`/api/game/${this.gameId}/history`);
            if (!response.ok) return;

            const data = await response.json();
            this.renderMoveHistory(data.history);
        } catch (error) {
            console.error('Error fetching move history:', error);
        }
    }

    renderMoveHistory(history) {
        this.moveList.innerHTML = '';

        const moves = {};
        history.forEach(move => {
            if (!moves[move.number]) {
                moves[move.number] = {};
            }
            moves[move.number][move.color] = move.san;
        });

        Object.keys(moves).forEach(moveNum => {
            const moveItem = document.createElement('div');
            moveItem.className = 'move-item';

            const moveNumber = document.createElement('span');
            moveNumber.className = 'move-number';
            moveNumber.textContent = moveNum + '.';

            const whiteMove = document.createElement('span');
            whiteMove.className = 'move-white';
            whiteMove.textContent = moves[moveNum].white || '';

            const blackMove = document.createElement('span');
            blackMove.className = 'move-black';
            blackMove.textContent = moves[moveNum].black || '';

            moveItem.appendChild(moveNumber);
            moveItem.appendChild(whiteMove);
            moveItem.appendChild(blackMove);

            this.moveList.appendChild(moveItem);
        });

        this.moveList.scrollTop = this.moveList.scrollHeight;
    }

    async resignGame() {
        if (!confirm('Are you sure you want to resign?')) {
            return;
        }

        try {
            const response = await fetch(`/api/game/${this.gameId}/resign`, {
                method: 'POST'
            });

            if (response.ok) {
                this.showGameOver('You resigned. AI wins!');
            }
        } catch (error) {
            console.error('Error resigning:', error);
        }
    }

    showGameOver(message) {
        this.gameOverMessage.textContent = message;
        this.gameOverModal.style.display = 'flex';
        this.isPlayerTurn = false;
    }

    resetToSetup() {
        this.gameOverModal.style.display = 'none';
        this.gamePanel.style.display = 'none';
        this.setupPanel.style.display = 'block';

        this.gameId = null;
        this.game = new Chess();
        this.isPlayerTurn = true;

        if (this.board) {
            this.board.destroy();
            this.board = null;
        }
    }

    showLoading(show) {
        this.loadingOverlay.style.display = show ? 'flex' : 'none';
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new ChessAI();
});