# import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE_bmt, TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '1,3'
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.set_device(args.local_rank)
dist.init_process_group(backend='nccl')
world_size = torch.distributed.get_world_size()
rank = torch.distributed.get_rank()
# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "/yinxr/liangshihao/OpenKE/benchmarks/FB15K237/", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader("/yinxr/liangshihao/OpenKE/benchmarks/FB15K237/", "link")

# define the model
transe_bmt = TransE_bmt(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200, 
	p_norm = 1, 
	norm_flag = True,
	local_rank=args.local_rank,
	world_size=world_size)

# transe = TransE(
# 	ent_tot = train_dataloader.get_ent_tot(),
# 	rel_tot = train_dataloader.get_rel_tot(),
# 	dim = 200, 
# 	p_norm = 1, 
# 	norm_flag = True)

# define the loss function
model_bmt = NegativeSampling(
	model = transe_bmt, 
	loss = MarginLoss(margin = 5.0).to(args.local_rank),
	batch_size = train_dataloader.get_batch_size()
).to(args.local_rank)
# model_bmt = DDP(model_bmt, device_ids=[args.local_rank])
# print(model_bmt.module.model.ent_embeddings.weight.data)
# x = input()
# model = NegativeSampling(
# 	model = transe, 
# 	loss = MarginLoss(margin = 5.0),
# 	batch_size = train_dataloader.get_batch_size()
# )

# train the model
trainer_bmt = Trainer(model = model_bmt, data_loader = train_dataloader, train_times = 1000, alpha = 1.0, use_gpu = True, local_rank=args.local_rank)
# trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1, alpha = 1.0, use_gpu = True)
trainer_bmt.run()
# trainer.run()

transe_bmt.save_checkpoint('./checkpoint/transe_dim200_node_{}.ckpt'.format(str(args.local_rank)))

# test the model
transe_bmt.load_checkpoint('./checkpoint/transe_dim200_node_{}.ckpt'.format(str(args.local_rank)))
tester = Tester(model = transe_bmt, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)