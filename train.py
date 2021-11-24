import sys
import argparse
import random
import torch
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from lit_transformer_model import Seq2SeqTransformer

def parse_args():
	parser = argparse.ArgumentParser("data generation tool")
	parser.add_argument('--learning_rate', type=float, default=0.0001)
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--nhead', type=int, default=8)
	parser.add_argument('--num_encoder_layers', type=int, default=6)
	parser.add_argument('--num_decoder_layers', type=int, default=6)
	parser.add_argument('--dim_feedforward', type=int, default=2048)
	args = parser.parse_args(sys.argv[1:])
	if len(sys.argv) < 1:
		parser.print_help()
	return args

if __name__ == "__main__":

	args = parse_args()
	print("args", args)
	
	logger = TensorBoardLogger("~/pytorch_logs", name="Seq2SeqTransformer")
	
	torch.backends.cudnn.deterministic = True
	torch.set_printoptions(profile="full")
	torch.manual_seed(0)
	random.seed(0)
	np.random.seed(0)	
	
	model = Seq2SeqTransformer(batch_size = args.batch_size, 
							dim_feedforward = args.dim_feedforward, 
							learning_rate = args.learning_rate,
							nhead = args.nhead,
							num_decoder_layers = args.num_decoder_layers,
							num_encoder_layers = args.num_encoder_layers)
							
	for p in model.parameters():
		if p.dim() > 1:
			torch.nn.init.xavier_uniform_(p)

	lr_monitor = LearningRateMonitor(logging_interval='step')
	trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=50, callbacks=[lr_monitor])
	trainer.fit(model)
