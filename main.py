import logging

# import coloredlogs
import torch

from Coach import Coach
from yacht.YachtGame import YachtGame as Game
from yacht.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

# coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s[%(process)d] %(levelname)s %(message)s')

args = dotdict({
    'numIters': 1000,
    'numEps': 20,              # More episodes for better sampling
    'tempThreshold': 15,       # Higher threshold for more exploration in early game
    'updateThreshold': 0.55,   # Lower threshold for more frequent updates
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,         # More MCTS simulations for better move evaluation
    'arenaCompare': 10,        # More games for better model comparison
    'cpuct': 1.5,              # Higher exploration constant

    'checkpoint': './temp/yacht/',
    'load_model': False,
    'load_folder_file': ('./temp/yacht/', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 5,  # Reduced memory usage

    # NN args - Optimized for yacht game learning
    'lr': 2e-3,                # Higher learning rate for faster adaptation
    # Stronger regularization to prevent overfitting to bad strategies
    'weight_decay': 1e-4,
    'epochs': 15,              # More epochs for better learning per iteration
    'batch_size': 512,         # Smaller batch size for more frequent updates
    'vloss_weight': 1.5,       # Higher value loss weight - important for position evaluation
    'cuda': torch.cuda.is_available(),
    'hidden': 256,             # Reduced from 768 - much smaller model size
    'nblocks': 6,              # Reduced from 12 - keep model compact
    'dropout': 0.3,            # Higher dropout to prevent overfitting to aggressive bidding
})


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game()  # Yacht game doesn't need size parameter

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g, args)

    # Print device information
    device = "GPU (CUDA)" if args.cuda else "CPU"
    log.info('Training will run on: %s', device)
    if args.cuda:
        log.info('GPU name: %s', torch.cuda.get_device_name(0))
    else:
        log.info('CUDA not available - using CPU training (slower)')

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...',
                 args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(
            args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
