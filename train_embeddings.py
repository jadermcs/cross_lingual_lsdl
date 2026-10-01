"""Swap Qwen3's tokenizer for a BERT (WordPiece) tokenizer, make it bidirectional, and train the
tied embeddings plus (optionally) the attention layers.

A single new embedding matrix (|V_bert| x d) is used both as input embeddings and as the LM head
(tied). Since the attention was pretrained causally, `--trainable attention` (default) also trains
the self-attention blocks (q/k/v/o projections and q/k norms) at a lower learning rate so they can
adapt to seeing the full sequence; `--trainable all` trains the whole model.

With bidirectional attention next-token prediction is trivial, so training uses masked tokens:
  mntp: masked token i is predicted from the output at position i-1 (LLM2Vec). This keeps the
        frozen body's pretrained "output predicts the next token" alignment.
  mlm:  masked token i is predicted from the output at position i (BERT).
"""

import argparse
import itertools

import torch
from bidirectional import make_bidirectional
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--tokenizer", default="google-bert/bert-base-uncased")
    p.add_argument("--init", choices=["mean", "random"], default="mean",
                   help="mean: average the Qwen embeddings of each BERT token's Qwen sub-tokens")
    p.add_argument("--objective", choices=["mntp", "mlm"], default="mntp")
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--dataset", default="Salesforce/wikitext")
    p.add_argument("--dataset_config", default="wikitext-103-raw-v1")
    p.add_argument("--split", default="train")
    p.add_argument("--text_column", default="text")
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--output_dir", default="checkpoints/qwen3-0.6b-bert-emb-bidir")
    p.add_argument("--trainable", choices=["embeddings", "attention", "all"], default="attention")
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate for the new embeddings")
    p.add_argument("--body_lr", type=float, default=5e-5, help="learning rate for pretrained transformer weights")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=10_000)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--report_to", default="none")
    p.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                   help="auto: bf16 on Ampere+ (sm>=80), fp16 on older GPUs like V100, fp32 on CPU")
    return p.parse_args()


@torch.no_grad()
def build_embeddings(old_emb, old_tok, new_tok, init):
    """Return a new (|V_new| x d) embedding matrix, optionally initialized from the old one."""
    d = old_emb.shape[1]
    new_emb = torch.empty(len(new_tok), d, dtype=torch.float32)
    std = old_emb.float().std().item()
    new_emb.normal_(0.0, std)
    if init == "random":
        return new_emb

    old_f = old_emb.float()
    mean_vec = old_f.mean(0)
    special = set(new_tok.all_special_tokens)
    hits = 0
    for token, idx in new_tok.get_vocab().items():
        if token in special:
            new_emb[idx] = mean_vec
            continue
        # WordPiece: "##x" continues a word; anything else starts one (byte-BPE marks that with a space).
        text = token[2:] if token.startswith("##") else " " + token
        ids = old_tok(text, add_special_tokens=False)["input_ids"]
        if ids:
            new_emb[idx] = old_f[ids].mean(0)
            hits += 1
    print(f"Initialized {hits}/{len(new_tok)} embeddings from source sub-token means")
    return new_emb


def swap_vocab(model, old_tok, new_tok, init):
    old_emb = model.get_input_embeddings().weight
    weight = build_embeddings(old_emb, old_tok, new_tok, init).to(old_emb.dtype)

    emb = torch.nn.Embedding(weight.shape[0], weight.shape[1], padding_idx=new_tok.pad_token_id)
    emb.weight = torch.nn.Parameter(weight)
    model.set_input_embeddings(emb)

    lm_head = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    lm_head.weight = emb.weight  # tie
    model.set_output_embeddings(lm_head)

    cfg = model.config
    cfg.vocab_size = len(new_tok)
    cfg.tie_word_embeddings = True
    cfg.pad_token_id = new_tok.pad_token_id
    cfg.bos_token_id = new_tok.cls_token_id
    cfg.eos_token_id = new_tok.sep_token_id
    if model.generation_config is not None:
        model.generation_config.pad_token_id = cfg.pad_token_id
        model.generation_config.bos_token_id = cfg.bos_token_id
        model.generation_config.eos_token_id = cfg.eos_token_id


def set_trainable(model, trainable):
    """Freeze everything except the requested parts; return the trainable body (non-embedding) params."""
    emb = model.get_input_embeddings().weight
    assert model.get_output_embeddings().weight is emb
    body = []
    for name, p in model.named_parameters():
        train = p is emb or trainable == "all" or (trainable == "attention" and ".self_attn." in name)
        p.requires_grad_(train)
        if train and p is not emb:
            body.append(p)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_train:,} / {n_total:,} ({100 * n_train / n_total:.1f}%)")
    return body


def build_dataset(args, tok):
    ds = load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=args.streaming)
    ds = ds.filter(lambda ex: bool(ex[args.text_column] and ex[args.text_column].strip()))
    sep = tok.sep_token_id

    def tokenize(batch):
        enc = tok(batch[args.text_column], add_special_tokens=False)["input_ids"]
        return {"input_ids": [ids + [sep] for ids in enc]}

    def pack(batch):
        flat = list(itertools.chain.from_iterable(batch["input_ids"]))
        n = len(flat) // args.seq_len * args.seq_len
        return {"input_ids": [flat[i:i + args.seq_len] for i in range(0, n, args.seq_len)]}

    cols = ds.column_names if not args.streaming else list(next(iter(ds)).keys())
    ds = ds.map(tokenize, batched=True, remove_columns=cols)
    ds = ds.map(pack, batched=True, batch_size=1000)
    return ds


class MaskedCollator(DataCollatorForLanguageModeling):
    """BERT-style masking (80% [MASK] / 10% random / 10% kept), with labels laid out for the
    causal-LM loss, which predicts labels[i + 1] from position i."""

    def __init__(self, tokenizer, objective, mask_prob):
        super().__init__(tokenizer, mlm=True, mlm_probability=mask_prob)
        self.objective = objective

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        if self.objective == "mlm":
            # Shift right so the loss's left shift lands each label back on its own position.
            labels = batch["labels"]
            batch["labels"] = torch.cat([torch.full_like(labels[:, :1], -100), labels[:, :-1]], dim=1)
        return batch


def resolve_precision(precision):
    if precision != "auto":
        return precision
    if not torch.cuda.is_available():
        return "fp32"
    # torch.cuda.is_bf16_supported() can report True on V100 via slow emulation, so check the arch.
    return "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16"


def main():
    args = parse_args()
    old_tok = AutoTokenizer.from_pretrained(args.model)
    # Drop BERT's token_type_ids: Qwen doesn't accept them (generate() errors out).
    new_tok = AutoTokenizer.from_pretrained(args.tokenizer, model_input_names=["input_ids", "attention_mask"])
    new_tok.model_max_length = args.seq_len

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    swap_vocab(model, old_tok, new_tok, args.init)
    make_bidirectional(model)
    model.config.training_objective = args.objective  # tells inference which position predicts a mask
    body_params = set_trainable(model, args.trainable)
    param_groups = [{"params": [model.get_input_embeddings().weight], "lr": args.lr}]
    if body_params:
        param_groups.append({"params": body_params, "lr": args.body_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    train_ds = build_dataset(args, new_tok)
    precision = resolve_precision(args.precision)
    print(f"Precision: {precision} (master weights fp32)")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=precision == "bf16",
        fp16=precision == "fp16",
        dataloader_num_workers=2,
        report_to=args.report_to,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=MaskedCollator(new_tok, args.objective, args.mask_prob),
        processing_class=new_tok,
        optimizers=(optimizer, None),  # scheduler is built by Trainer and scales both groups
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    new_tok.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
