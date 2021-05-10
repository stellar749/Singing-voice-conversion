import os,sys
import argparse
import logging
from Config import set_config
from Dataset import DataProcessing

#Set up
config = set_config('config.ini')
logging.basicConfig(filename=config.get('logging', 'logfile'), \
    level=logging.getLevelName(config.get('logging', 'loglevel')), \
    filemode='a', format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',\
    datefmt='%a, %d %b %Y %H:%M:%S')

parser = argparse.ArgumentParser(description="Neural network for vocal and music splitting")
parser.add_argument("--mode", default="train", type=str, help="Mode in which the script is run (train/separate).")
parser.add_argument("--weights", default="network.weights", type=str, help="File containing the weights to be used with the neural network. Will be created if it doesn't exist. Required for separation. Default is network.weights.")
parser.add_argument("--epochs", default=1, type=int, help="How many times will the network go over the data. default - 1. (requires --mode=train)")
parser.add_argument("--inputfile", default="mixture.wav", type=str, help="Name of the file from which to extract vocals. (requires --mode=separate)")
parser.add_argument("--output", default="vocals.wav", type=str, help="Name of the file to which the vocals will be written to. (requires --mode=separate)")
parser.add_argument("--save_accompaniment", default="false", type=str, help="If set to true, the accompaniment will also be saved as a separate file (requires --mode=separate)")
args = parser.parse_args()

if args.mode == "train":
    logging.info("Preparing to train a model...")
    logging.info("Saving weights...")
    data = DataProcessing(logging, config)
    data.load(config.get("data", "traindir"))
    data.stft()
    data.split()
elif args.mode == "separate":
    logging.info("Nothing")
else:
    logging.critical("Invalid action - %s", args.mode)
    sys.exit(12)