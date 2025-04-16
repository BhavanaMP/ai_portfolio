import argparse
import os
import pickle
from contextlib import nullcontext
from dataclasses import dataclass

# Import and configure the logger
import logging
import logger_config
# Call this at the start of your main script to set up logging
logger_config.setup_logging()
logger = logging.getLogger(__name__)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from data.prepare_data import read_input
from training.model import GPT


@dataclass
class ModelArgs:
    # Set the Hyperparameters - add these to the args namespace if you want to run in notebook or args as dict with keys doesntwork..we need a class
    n_ctx: int = 256
    embed_dim: int = 384 
    num_heads: int = 6
    num_layers: int = 6
    bs: int = 62 
    lr: float = 3e-4
    num_epochs: int = 5000
    eval_iters: int = 200
    eval_interval: int = 500
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # We can pass args as class variables if we are not taking from user. just pass args: ModelArgs in main() args

def ddp_setup(rank: int, world_size: int, backend, init_method):
    """
    Args:
    rank: Unique identifier of each process. Each GPU has one process
    world_size: Total No. of processes or GPUs
    backend: Backend to use for distributed communcation among GPUs
    (nccl - NVIDIA Collective Communications Library)
    """
    torch.cuda.set_device(rank)
    init_process_group(backend=backend, init_method=init_method, world_size=world_size, rank=rank, timeout=timedelta(seconds=6000)) #100 minutes
    dist.barrier()

class Trainer:
    def __init__(self, args, device) -> None:
        self.args = args
        self.device = device
        # Process the input text
        self.process_input(args.input_data_path)
        # Generate Splits
        self.chars_tokenized = torch.tensor(self.encode(self.chars), dtype=torch.long)
        train_percentage = int(0.9 * len(list(self.chars_tokenized)))
        self.train_split = self.chars_tokenized[:train_percentage]
        self.val_split = self.chars_tokenized[train_percentage:]
        print(f"Train split: {len(self.train_split)}")
        print(f"Val split: {len(self.val_split)}")
        
        # Initialize the Language Model
        self.model = GPT(self.num_vocab, self.args)  # Fix the args
        self.model.to(self.device)
        # print the number of parameters in the model
        print(sum(param.numel() for param in self.model.parameters()) / 1e6, "M parameters")
        # Optimizer
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.args.lr)
        
        # Checkpoint saving logic
        self.best_loss = torch.
    
    def process_input(self, inp_path):
        # Read the input data
        self.chars = read_input(path=inp_path)
        logger.debug("\nFirst 100 chars in the input file: %s", self.chars[:100])
        print(f"Total chars in the input data: {len(self.chars)}")

        # Unique Total Vocabulary
        self.vocab = sorted(list(set(self.chars)))
        self.num_vocab = len(self.vocab)
        print(f"Vocabulary: {self.vocab}, \nVocab Length: {self.num_vocab}")

        # Simple Tokenizer - Mapping  of chars to integers
        # Simple Tokenization of chars. Open AI uses TikToken which is a fast BPE/sub-word tokenizer
        self.char2tok = {c: idx for idx, c in enumerate(self.vocab)}
        self.tok2char = {idx: c for idx, c in enumerate(self.vocab)}

        # Encode a list of chars into list of numbers i.e tokens
        self.encode = lambda chars: [self.char2tok[c] for c in chars]
        # Decode the list of numbers into the chars
        self.decode = lambda tokens: "".join([self.tok2char[idx] for idx in tokens])

        logger.debug("Simpler Tokenizer - Chars to Tokens: %s", self.char2tok)
        logger.debug("Simple Tokenizer - Tokens to Chars: %s", self.tok2char)
        logger.debug("Encoding 'Adam': %s", self.encode("Adam"))
        logger.debug("Decoding %s: %s", self.encode("Adam"), self.decode(self.encode("Adam")))

    def get_batch(self, split="train"):
        torch.manual_seed(1234)
        n_ctx = self.args.n_ctx
        bs = self.args.bs
        data = self.train_split if split == "train" else self.val_split
        idxs = torch.randint(high=len(data)-n_ctx, size=(bs,))  # For len(data)=1003854, Generate a 1D tensor with bs elements. Each element is a random integer in the range [0, 1003846)
        X = torch.stack([data[i:i+n_ctx] for i in idxs])  # bs, n_ctx
        Y = torch.stack([data[i+1:i+n_ctx+1] for i in idxs])  # bs, n_ctx # right shifted targets
        return X, Y

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        # Put the Model in Evaluation Mode
        self.model.eval()
        logger.info("Model is in Eval Mode.")
        for split in ["train", "val"]:
            losses = torch.zeros(self.args.eval_iters)
            for k in range(self.args.eval_iters):
                X, Y = self.get_batch(split)
                _, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        # Put the model back in train mode
        self.model.train()
        logger.info("Model is back in Train Mode.")
        return out

    def train(self):
        # Training Loop
        self.model.train()
        for epoch in range(self.args.num_epochs):
            
            # Sample a batch of training data
            X, Y = self.get_batch(split="train")
            X, Y = X.to(self.device), Y.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            logits, loss = self.model(X, Y)
            loss.backward()
            self.optimizer.step()

            # every once in a while evaluate the loss on train and val sets
            if epoch % self.args.eval_interval == 0 or epoch == self.args.num_epochs - 1:
                losses = self.estimate_loss()
                print(f"Step {epoch}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}")
                

def main(device, args):
    logger.info("Starting main function")
    
    if args.is_distributed:
        # Set up and Initialize the process group
        print(f"[INFO]: Trying distributed setup with rank {device}")
        ddp_setup(rank=device, world_size=args.world_size, backend=args.backend, init_method=args.init_method)
        print("[INFO]: Distributed group initialized")
    elif torch.cuda.is_available() and args.world_size == 1:
        device = 0  # sets device=cuda:0 explicilty if gpu exist & only 1 available
    else:
        device = "cpu"
    
    # Training
    # Note: For usual encoder decoder transformer, unlike GPT, training is teacher forching(one fwd pass whole sentence, right shifting) and inference is auto regressive
    torch.manual_seed(1337)

    trainer = Trainer(args=args, device=device)
    trainer.train()
    
    # Inference - Generate Text from the trained model
    input_ctx = torch.zeros((1, 1), dtype=torch.long, device=device)  # bs, n_ctx # Note bs represents how may sentences we are sending for each forward pass.. Here we are sending only 1 sentence. There ll b no interaction in bs dim.
    output = trainer.model.generate_text(input_ctx, max_num_tokens=10000)[0].tolist()  # [0] as we are accessing bs and we having only 1 in bs for generation, tolist converts tensor containing int contexts to list
    # print(trainer.decode(output))  # Uncomment to print in the terminal

    # Write to a file
    with open("generation_out.txt", "w", encoding="utf-8") as f:
        f.write(trainer.decode(output))

    logger.info("Finished main function")
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_path", type=str, default="data/shakespeare.txt")
    parser.add_argument("--n_ctx", type=int, default=256, help="Maximum Sequence / context length? i.e maximum tokens in the sequence")
    parser.add_argument("--embed_dim", type=int, default=384, help="Embedding dimension of each token in the context")  # usually 768 for original tranforms, or 1024, 1048, 2048
    parser.add_argument("--num_heads", type=int, default=6, help="No of heads for multi-headed self attention")
    parser.add_argument("--num_layers", type=int, default=6, help="No. of blocks of MSA + FFN to run. i,e num of layers to MSA before passing to Linear Head")
    parser.add_argument("--bs", type=int, default=64, help="how many independent sequences/sentences/chars will we process in parallel?")
    parser.add_argument("--lr", type=int, default=3e-4, help="Learning Rate")
    parser.add_argument("--num_epochs", type=int, default=5000, help="Num of epochs to train the model for")
    parser.add_argument("--eval_interval", type=int, default=500, help="Epoch interval to estimate the proper evaluation loss")
    parser.add_argument("--eval_iters", type=int, default=200, help="How many iters to properly estimating the Mean Train and Val Evaluation loss")
    parser.add_argument("--is_train", type=bool, default=True)
    parser.add_argument("--out_dir", type=str, default="out/generation_out.txt")
    args = parser.parse_args()
    
    # Device
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        if world_size > 1:
            # initiate distributed training
            try:
                mp.set_start_method("spawn", force=True)
                print("[INFO]: Spawned")
            except RuntimeError:
                pass
            processes = []
            for rank in range(world_size):
                p = Process(target=main, args=(rank, args))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        elif world_size == 1:
            device = "cuda"
    else:
        device = "cpu"
        
    # Call the main
    main(device, args)

# python -m training.train to run this file as script
