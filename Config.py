import os


basedir, _ = os.path.split(os.path.abspath(__file__))
basedir = os.path.join(basedir, 'data')

CHECKPOINT_DIR = os.path.join(basedir, 'checkpoint')
RESULTS_DIR = os.path.join(basedir, 'results')
DATASET_DIR = os.path.join(basedir, 'datasets')

BATCH_SIZE = 128
TEST_SET_SIZE = 1000000000
NUM_TRANS = 8

INPUT_CH = 0
INPUT_DIM = 0
INPUT_PAD = 0
INPUT_DIM_PADDED = 0
