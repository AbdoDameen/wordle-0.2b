"""wordle02b: a 0.2B-parameter transformer that learns to play Wordle."""
__version__ = "0.1.0"

from .baseline import EntropySolver, build_pattern_matrix_cached
from .evaluate import play_games, play_games_baseline, summarize
from .game import WordleGame, feedback
from .model import GPT, GPTConfig
from .words import Vocabulary, load_word_lists
