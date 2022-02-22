# coding: utf-8
import argparse
import math
import time

import torch
import torch.nn as nn

import data
import model

parser = argparse.ArgumentParser(
    description="PyTorch Wikitext-2 RNN/LSTM/GRU Language Model"
)
parser.add_argument(
    "--model",
    type=str,
    default="RNN_TANH",
    help="type of network (RNN_TANH, RNN_RELU, LSTM, GRU)",
)
parser.add_argument(
    "--hidden_dim", type=int, default=100, help="number of hidden units per layer"
)
parser.add_argument("--lr", type=float, default=20, help="initial learning rate")
parser.add_argument("--clip", type=float, default=0.25, help="gradient clipping")
parser.add_argument("--epochs", type=int, default=40, help="upper epoch limit")
parser.add_argument(
    "--batch_size", type=int, default=20, metavar="N", help="batch size"
)
parser.add_argument("--seq_len", type=int, default=35, help="sequence length")
parser.add_argument("--seed", type=int, default=1111, help="random seed")
parser.add_argument("--cuda", action="store_true", help="use CUDA")
parser.add_argument(
    "--log-interval", type=int, default=200, metavar="N", help="report interval"
)
parser.add_argument(
    "--save", type=str, default="model.pt", help="path to save the final model"
)
parser.add_argument(
    "--dry-run", action="store_true", help="verify the code and the model"
)
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################################
# Load data
###############################################################################
corpus = data.Corpus()

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).contiguous()
    print("data shape is ", data.shape)
    return data.to(device)


eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

model = model.FNNModel(
    token_num=len(corpus.dictionary),
    seq_len=args.seq_len,
    hidden_dim=args.hidden_dim,
    use_direct_connection=True,
).to(device)

criterion = nn.NLLLoss()

###############################################################################
# Training code
###############################################################################


# get_batch subdivides the source data into chunks of length args.seq_len.
# If source is equal to the example output of the batchify function, with
# a seq_len-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.


def get_batch(source, i):
    seq_len = min(args.seq_len, source.shape[1] - 1 - i)
    data = source[:, i: i + seq_len]
    target = source[:, i + seq_len: i + 1 + seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i in range(0, data_source.size(1) - 1, args.seq_len):
            data, targets = get_batch(data_source, i)
            output = model(data)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.0
    start_time = time.time()
    for batch, i in enumerate(range(0, train_data.size(1) - 1, args.seq_len)):
        data, targets = get_batch(train_data, i)
        model.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | "
                "loss {:5.2f} | ppl {:8.2f}".format(
                    epoch,
                    batch,
                    len(train_data) // args.seq_len,
                    lr,
                    elapsed * 1000 / args.log_interval,
                    cur_loss,
                    math.exp(cur_loss),
                )
            )
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print("-" * 89)
        print(
            "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | "
            "valid ppl {:8.2f}".format(
                epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)
            )
        )
        print("-" * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, "wb") as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print("-" * 89)
    print("Exiting from training early")


# Run on test data.
test_loss = evaluate(test_data)
print("=" * 89)
print(
    "| End of training | test loss {:5.2f} | test ppl {:8.2f}".format(
        test_loss, math.exp(test_loss)
    )
)
print("=" * 89)
