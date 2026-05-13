"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm
import wandb

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import prepare_data, collate_fn, PAD_IDX, SOS_IDX, EOS_IDX
from lr_scheduler import NoamScheduler


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        log_probs = torch.log_softmax(logits, dim=-1)

        # create smoothed distribution
        smooth_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
        smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        smooth_dist[:, self.pad_idx] = 0

        # zero out loss for pad targets
        pad_mask = target == self.pad_idx
        smooth_dist[pad_mask] = 0

        # KL divergence
        loss = -(smooth_dist * log_probs).sum(dim=-1)
        # average over non-pad tokens
        non_pad = (~pad_mask).sum().clamp(min=1)
        return loss.sum() / non_pad


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).
    """
    model.train() if is_train else model.eval()
    total_loss   = 0.0
    total_tokens = 0

    pbar = tqdm(data_iter, desc=f"{'Train' if is_train else 'Val'} Epoch {epoch_num}", leave=False)

    for src, tgt in pbar:
        src, tgt = src.to(device), tgt.to(device)

        # teacher forcing: decoder input = tgt[:-1], target = tgt[1:]
        tgt_input  = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = make_src_mask(src, PAD_IDX).to(device)
        tgt_mask = make_tgt_mask(tgt_input, PAD_IDX).to(device)

        logits = model(src, tgt_input, src_mask, tgt_mask)

        # flatten for loss: (batch * tgt_len, vocab) and (batch * tgt_len)
        logits_flat = logits.contiguous().view(-1, logits.size(-1))
        target_flat = tgt_output.contiguous().view(-1)

        loss = loss_fn(logits_flat, target_flat)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            # gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        # count non-pad tokens
        n_tokens = (tgt_output != PAD_IDX).sum().item()
        total_loss   += loss.item() * n_tokens
        total_tokens += n_tokens

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_tokens, 1)


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.
    """
    model.eval()
    memory = model.encode(src, src_mask)
    ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys, PAD_IDX).to(device)
        logits = model.decode(memory, src_mask, ys, tgt_mask)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)
        if next_token.item() == end_symbol:
            break

    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with decode method.
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).
    """
    import math
    from collections import Counter

    def _ngrams(tokens, n):
        """Extract n-grams from a token list."""
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    def _corpus_bleu(candidates, references_list, max_n=4):
        """
        Compute corpus-level BLEU score (BLEU-4 by default).

        Args:
            candidates      : list of predicted token lists, e.g. [['a','b'], ...]
            references_list : list of reference lists, e.g. [[['a','b']], ...]
                              (each entry is a list of one or more reference token lists)
            max_n           : maximum n-gram order (default 4).

        Returns:
            BLEU score in [0, 1].
        """
        clipped_counts = [0] * max_n
        total_counts   = [0] * max_n
        bp_c = 0  # candidate length
        bp_r = 0  # effective reference length

        for cand, refs in zip(candidates, references_list):
            bp_c += len(cand)
            # closest reference length
            ref_lens = [len(ref) for ref in refs]
            closest = min(ref_lens, key=lambda r: (abs(r - len(cand)), r))
            bp_r += closest

            for n in range(1, max_n + 1):
                cand_ngrams = Counter(_ngrams(cand, n))
                # max counts across all references
                max_ref_ngrams = Counter()
                for ref in refs:
                    ref_ngrams = Counter(_ngrams(ref, n))
                    for ng, cnt in ref_ngrams.items():
                        max_ref_ngrams[ng] = max(max_ref_ngrams[ng], cnt)

                clipped = {ng: min(cnt, max_ref_ngrams[ng])
                           for ng, cnt in cand_ngrams.items()}
                clipped_counts[n - 1] += sum(clipped.values())
                total_counts[n - 1]   += sum(cand_ngrams.values())

        # compute modified precisions (log domain)
        log_precision = 0.0
        for n in range(max_n):
            if total_counts[n] == 0 or clipped_counts[n] == 0:
                return 0.0  # if any n-gram precision is zero, BLEU is zero
            log_precision += math.log(clipped_counts[n] / total_counts[n])
        log_precision /= max_n

        # brevity penalty
        if bp_c >= bp_r:
            bp = 1.0
        else:
            bp = math.exp(1 - bp_r / bp_c)

        return bp * math.exp(log_precision)

    model.eval()
    all_refs  = []
    all_preds = []

    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc="BLEU Eval", leave=False):
            src = src.to(device)
            tgt = tgt.to(device)

            for i in range(src.size(0)):
                src_i = src[i].unsqueeze(0)
                src_mask_i = make_src_mask(src_i, PAD_IDX).to(device)

                pred_tokens = greedy_decode(
                    model, src_i, src_mask_i,
                    max_len, SOS_IDX, EOS_IDX, device,
                )

                # decode predicted indices to words (strip specials)
                pred_indices = pred_tokens.squeeze(0).tolist()
                pred_words = tgt_vocab.decode(pred_indices, strip_special=True)

                # decode reference indices to words (strip specials)
                ref_indices = tgt[i].tolist()
                ref_words = tgt_vocab.decode(ref_indices, strip_special=True)

                all_preds.append(pred_words)
                all_refs.append([ref_words])  # list of references per sentence

    bleu = _corpus_bleu(all_preds, all_refs) * 100.0
    return bleu


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    The autograder will call load_checkpoint to restore your model.
    Do NOT change the keys in the saved dict.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').
    """
    torch.save({
        'epoch':                epoch,
        'model_state_dict':    model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'model_config':        model.config,
    }, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).
    """
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    return ckpt['epoch']


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(train_loader, model, loss_fn,
                             optimizer, scheduler, epoch, is_train=True)
                   run_epoch(val_loader, model, loss_fn,
                             None, None, epoch, is_train=False)
                   save_checkpoint(model, optimizer, scheduler, epoch)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test_bleu': bleu})
    """
    # ── parse arguments ──
    parser = argparse.ArgumentParser(description="DA6401 Assignment 3 Transformer Training")
    parser.add_argument('--d_model',      type=int,   default=256)
    parser.add_argument('--n_layers',     type=int,   default=3)
    parser.add_argument('--num_heads',    type=int,   default=8)
    parser.add_argument('--d_ff',         type=int,   default=512)
    parser.add_argument('--dropout',      type=float, default=0.1)
    parser.add_argument('--epochs',       type=int,   default=30)
    parser.add_argument('--batch_size',   type=int,   default=128)
    parser.add_argument('--warmup_steps', type=int,   default=4000)
    parser.add_argument('--smoothing',    type=float, default=0.1)
    parser.add_argument('--min_freq',     type=int,   default=2)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--wandb_project',  type=str, default='da6401-a3')
    parser.add_argument('--run_name',       type=str, default='')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── 1. prepare data ──
    print("Loading data and building vocabularies...")
    train_ds, val_ds, test_ds, src_vocab, tgt_vocab, de_tok, en_tok = prepare_data(min_freq=args.min_freq)

    print(f"  src_vocab: {len(src_vocab)} | tgt_vocab: {len(tgt_vocab)}")
    print(f"  train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}")

    # ── 2. dataloaders ──
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    # ── 3. model ──
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        N=args.n_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        pad_idx=PAD_IDX,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params:,}")

    # ── 4. optimizer & scheduler ──
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0,
        betas=(0.9, 0.98), eps=1e-9,
    )
    scheduler = NoamScheduler(optimizer, d_model=args.d_model, warmup_steps=args.warmup_steps)

    # ── 5. loss ──
    loss_fn = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, smoothing=args.smoothing)

    # ── 6. W&B init ──
    config = {
        'd_model': args.d_model, 'n_layers': args.n_layers,
        'num_heads': args.num_heads, 'd_ff': args.d_ff,
        'dropout': args.dropout, 'epochs': args.epochs,
        'batch_size': args.batch_size, 'warmup_steps': args.warmup_steps,
        'smoothing': args.smoothing, 'min_freq': args.min_freq,
        'src_vocab_size': len(src_vocab), 'tgt_vocab_size': len(tgt_vocab),
        'total_params': total_params,
    }

    wandb.init(
        project=args.wandb_project,
        name=args.run_name or f"d{args.d_model}_N{args.n_layers}_h{args.num_heads}",
        config=config,
    )
    wandb.watch(model, log='gradients', log_freq=100)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')

    # ── 7. training loop ──
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler, epoch, True, device)
        val_loss   = run_epoch(val_loader,   model, loss_fn, None,      None,      epoch, False, device)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6e}")

        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss,
            'val/loss':   val_loss,
            'lr': current_lr,
        })

        # save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, scheduler, epoch, best_path)
            print(f"  Saved best checkpoint -> {best_path}")

        # save periodic checkpoint
        ckpt_path = os.path.join(args.checkpoint_dir, 'checkpoint.pt')
        save_checkpoint(model, optimizer, scheduler, epoch, ckpt_path)

    # ── 8. final BLEU evaluation ──
    print("\nLoading best checkpoint for final evaluation...")
    best_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
    if os.path.exists(best_path):
        load_checkpoint(best_path, model)

    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device)
    print(f"\nTest BLEU: {bleu:.2f}")
    wandb.log({'test_bleu': bleu})
    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    run_training_experiment()
