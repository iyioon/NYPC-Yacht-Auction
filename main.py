import logging

import coloredlogs

from Coach import Coach
from yacht.Game import Game
from yacht.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 50,             # Reduced for initial testing
    # Number of complete self-play games to simulate during a new iteration.
    'numEps': 20,
    'tempThreshold': 15,        # Temperature threshold for MCTS
    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'updateThreshold': 0.6,
    # Number of game examples to train the neural networks (reduced for yacht game)
    'maxlenOfQueue': 50000,
    # Number of MCTS simulations per move (reduced for faster training)
    'numMCTSSims': 10,
    # Number of games to play during arena play to determine if new net will be accepted.
    'arenaCompare': 20,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp/', 'best.pth.tar'),  # Fixed path
    'numItersForTrainExamplesHistory': 10,  # Reduced for yacht game

})


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game()  # Yacht game doesn't need size parameter

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

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
