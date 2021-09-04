import sys
import argparse
import io
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence

torch.backends.cudnn.deterministic = True
torch.set_printoptions(profile="full")

import pandas as pd
import numpy as np

from torch.optim.lr_scheduler import MultiStepLR
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

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
	
class ReverseStringsDataset(Dataset):
	def __init__(self, data):
		super(self.__class__, self).__init__()
		self.inputs = data[0].values
		self.outputs = data[1].values

	def __len__(self):
		return len(self.inputs)

	def __getitem__(self, idx):
		return self.inputs[idx].strip(), self.outputs[idx].strip()

class PositionalEncoding(nn.Module):
	def __init__(self,
				 emb_size: int,
				 dropout: float = 0.1,
				 maxlen: int = 500):
		super(PositionalEncoding, self).__init__()
		den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
		pos = torch.arange(0, maxlen).reshape(maxlen, 1)
		pos_embedding = torch.zeros((maxlen, emb_size))
		pos_embedding[:, 0::2] = torch.sin(pos * den)
		pos_embedding[:, 1::2] = torch.cos(pos * den)
		pos_embedding = pos_embedding.unsqueeze(-2)
		self.dropout = nn.Dropout(dropout)
		self.emb_size = emb_size
		self.register_buffer('pos_embedding', pos_embedding)

	def forward(self, token: Tensor):
		token_embedding = torch.nn.functional.one_hot(token, num_classes=self.emb_size)
		return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

def yield_tokens(dataset):
	for i in range(len(dataset)):
		yield dataset[i][0].strip().split() + dataset[i][1].strip().split()

def vocab_func(vocab):
	def func(tok_iter):
		return [vocab[tok] for tok in tok_iter]
	return func

def sequential_transforms(*transforms):
	def func(txt_input):
		for transform in transforms:
			txt_input = transform(txt_input)
		return txt_input
	return func

class Seq2SeqTransformer(pl.LightningModule):
	def __init__(self, batch_size = 32, 
						dim_feedforward = 2048, 
						learning_rate = 0.0001,
						nhead = 8,
						num_decoder_layers = 6,
						num_encoder_layers = 6):
		
		super(Seq2SeqTransformer, self).__init__()
		self.learning_rate = learning_rate

		train_data_file = './data/rev_train_256.csv'
		val_data_file = './data/rev_val_256.csv'

		self.training_data = ReverseStringsDataset(pd.read_csv(train_data_file, header=None, sep=';'))
		self.val_data = ReverseStringsDataset(pd.read_csv(val_data_file, header=None, sep=';'))
		self.vocabulary = build_vocab_from_iterator(yield_tokens(self.training_data),
													specials=['<unk>', '<pad>', '<start>', '<eos>'],
													special_first=True)
		self.training_data = ReverseStringsDataset(pd.read_csv(train_data_file, header=None, sep=';'))
		self.start_idx = self.vocabulary['<start>']
		self.eos_idx = self.vocabulary['<eos>']
		self.pad_idx = self.vocabulary['<pad>']

		print("vocabulary", len(self.vocabulary), self.vocabulary.get_stoi())

		self.emb_size = 16
		self.batch_size = batch_size
		# self.batch_size = 4
		self.tgt_vocab_size = len(self.vocabulary)
		self.transforms = sequential_transforms(lambda x: x.split(), vocab_func(self.vocabulary), self.totensor(torch.long))
		self.transformer = nn.Transformer(d_model=self.emb_size, 
										dim_feedforward = dim_feedforward, 
										nhead = nhead,
										num_decoder_layers = num_decoder_layers,
										num_encoder_layers = num_encoder_layers)
		self.generator = nn.Linear(self.emb_size, self.tgt_vocab_size)
		self.positional_encoding = PositionalEncoding(self.emb_size)

		print("Seq2SeqTransformer")
		print("learning_rate", self.learning_rate)
		print("batch_size", self.batch_size)
		print("dim_feedforward", dim_feedforward)
		print("nhead", nhead)
		print("num_decoder_layers", num_decoder_layers)
		print("num_encoder_layers", num_encoder_layers)

	def collate_fn(self, batch):
		input_batch, output_batch = [], []
		for i, o in batch:
			input_batch.append(self.transforms(i))
			output_batch.append(self.transforms(o))
		input_batch = pad_sequence(input_batch, padding_value=self.pad_idx)
		output_batch = pad_sequence(output_batch, padding_value=self.pad_idx)
		return input_batch, output_batch

	def totensor(self, dtype):
		def func(ids_list):
			return torch.cat((torch.tensor([self.start_idx]).to(dtype),
							torch.tensor(ids_list).to(dtype),
							torch.tensor([self.eos_idx]).to(dtype)))
		return func

	def train_dataloader(self):
		train_dataloader = DataLoader(self.training_data, batch_size=self.batch_size, collate_fn=self.collate_fn)
		return train_dataloader

	def val_dataloader(self):
		val_dataloader = DataLoader(self.val_data, batch_size=self.batch_size, collate_fn=self.collate_fn)
		return val_dataloader

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
		return {
			"optimizer": optimizer,
			"lr_scheduler": {
				"scheduler": MultiStepLR(optimizer, milestones=[300, 400, 500], gamma=0.1),
				"name": "learning_rate_log",
			},
		}

	def forward(self,
				src: Tensor,
				tgt: Tensor,
				src_mask: Tensor,
				tgt_mask: Tensor,
				src_padding_mask: Tensor,
				tgt_padding_mask: Tensor,
				memory_key_padding_mask: Tensor):
		src_emb = self.positional_encoding(src)
		tgt_emb = self.positional_encoding(tgt)
		outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
								src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
		return self.generator(outs)

	def generate_square_subsequent_mask(self, sz):
		mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		return mask

	def create_mask(self, src, tgt):
		src_seq_len = src.shape[0]
		tgt_seq_len = tgt.shape[0]

		tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
		src_mask = torch.zeros((src_seq_len, src_seq_len), device=self.device).type(torch.bool)

		src_padding_mask = (src == self.pad_idx).transpose(0, 1)
		tgt_padding_mask = (tgt == self.pad_idx).transpose(0, 1)
		return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

	def greedy_decode(self, model, src, src_mask, max_len, start_symbol):
		src = src.to(self.device)
		src_mask = src_mask.to(self.device)

		memory = model.encode(src, src_mask)
		ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(self.device)
		for i in range(max_len-1):
			memory = memory.to(self.device)
			tgt_mask = (self.generate_square_subsequent_mask(ys.size(0))
						.type(torch.bool)).to(self.device)
			out = model.decode(ys, memory, tgt_mask)
			out = out.transpose(0, 1)
			prob = model.generator(out[:, -1])
			_, next_word = torch.max(prob, dim=1)
			next_word = next_word.item()

			ys = torch.cat([ys,
							torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
			if next_word == self.eos_idx:
				break
		return ys

	def translate(self, model: torch.nn.Module, src_sentence: str):
		model.eval()
		src = self.transforms(src_sentence).view(-1, 1)
		num_tokens = src.shape[0]
		src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
		tgt_tokens = self.greedy_decode(model, src, src_mask, max_len=num_tokens + 5, start_symbol=self.start_idx).flatten()
		return " ".join(self.vocabulary.lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<start>", "").replace("<eos>", "").strip()

	def encode(self, src: Tensor, src_mask: Tensor):
		return self.transformer.encoder(self.positional_encoding(src), src_mask)

	def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
		return self.transformer.decoder(self.positional_encoding(tgt), memory, tgt_mask)

	def training_step(self, batch, batch_idx):
		src, tgt = batch

		tgt_input = tgt[:-1, :]
		src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, tgt_input)
		logits = self.forward(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

		tgt_out = tgt[1:, :]
		loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1), ignore_index=self.pad_idx)
		
		self.log("train_loss", loss)

		return loss

	def validation_step(self, batch, batch_idx):
		src, tgt = batch

		tgt_input = tgt[:-1, :]
		src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, tgt_input)
		logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

		tgt_out = tgt[1:, :]
		loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1), ignore_index=self.pad_idx)

		self.log("val_loss", loss)

	def training_epoch_end(self, training_step_outputs):
		losses = 0
		for pred in training_step_outputs:
			losses += pred['loss'].item()
		losses /= len(training_step_outputs)
		print((f"epoch loss: {losses:.10f}"))

	# learning rate warm-up
	# def optimizer_step(
	# 	self,
	# 	epoch,
	# 	batch_idx,
	# 	optimizer,
	# 	optimizer_idx,
	# 	optimizer_closure,
	# 	on_tpu=False,
	# 	using_native_amp=False,
	# 	using_lbfgs=False,
	# ):
	# 	# skip the first 500 steps
	# 	if self.trainer.global_step < 500:
	# 		lr_scale = min(1.0, float(self.trainer.global_step + 1) / 500.0)
	# 		for pg in optimizer.param_groups:
	# 			pg["lr"] = lr_scale * self.learning_rate
	# 
	# 	# update params
	# 	optimizer.step(closure=optimizer_closure)

	def on_epoch_end(self):
	
		total = 0
		correct = 0
		for i in range(0, len(self.val_data)):
			input = self.val_data[i][0]
			output = self.val_data[i][1]
			prediction = self.translate(self, input)
			total += 1
			if output == prediction:
				correct += 1
			if i > 10:
				break
	
		accuracy = correct / total
		self.log("accuracy", accuracy, prog_bar=True)

if __name__ == "__main__":

	args = parse_args()
	print("args", args)
	
	logger = TensorBoardLogger("~/pytorch_logs", name="Seq2SeqTransformer")
	
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
			nn.init.xavier_uniform_(p)

	lr_monitor = LearningRateMonitor(logging_interval='step')
	trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=500, callbacks=[lr_monitor])
	trainer.fit(model)
