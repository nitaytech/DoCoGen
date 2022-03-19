from pathlib import Path
from project_utils.functions import count_num_cpu_gpu


NUM_CPU, NUM_GPU = count_num_cpu_gpu()
PRECISION = 32
DIM = 768
LOSS_IGNORE_ID = -100
UNK = 'Unknown'
SEP = '@@@'

T5_MODEL_NAME = 't5-base'
MAX_SEQ_LEN = 96
BATCH_SIZE = 48
NUM_WORKERS = NUM_CPU

PROJECT_NAME = "DoCoGen/"
PROJECT_DIR = Path.home() / "prod" / PROJECT_NAME
DATA_DIR = PROJECT_DIR / "data"
EXP_DIR = PROJECT_DIR / "experiments"