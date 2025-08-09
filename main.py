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
    'numEps': 15,              # Further reduced - much faster iterations
    'tempThreshold': 10,       # Reduced - faster transition to exploitation
    'updateThreshold': 0.6,    # Slightly higher - less frequent model updates
    'maxlenOfQueue': 200000,
    'numMCTSSims': 15,         # Much faster self-play
    'arenaCompare': 5,         # Much faster model comparison
    'cpuct': 1.0,

    'checkpoint': './temp/yacht/',
    'load_model': False,
    'load_folder_file': ('./temp/yacht/', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 5,  # Reduced memory usage

    # NN args - Balanced for quality and speed
    'lr': 1e-3,                # Reduced back to stable learning rate
    'weight_decay': 1e-5,
    'epochs': 10,              # Increased for better learning per iteration
    'batch_size': 1024,        # Reduced from 2048 - balance speed vs GPU usage
    'vloss_weight': 1.0,
    'cuda': torch.cuda.is_available(),
    'hidden': 512,             # Reduced from 1536 - faster training
    'nblocks': 8,              # Reduced from 16 - faster model
    'dropout': 0.2,
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
