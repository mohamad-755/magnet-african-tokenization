"""
MAGNET training loop. Wires together the existing model (src/model/) and
data pipeline (src/training/dataset.py, collate.py) — no architecture or
data-loading logic lives here.

Real training runs are expected to happen on GPU (e.g. Colab), not locally.
Usage:
    python -m src.training.train --data-root /path/to/DATA_ROOT

Resuming (--resume): only the step counter is restored from the
checkpoint, not the LR schedule's shape (PyTorch's LambdaLR can't
pickle the lambda closure that defines it — see build_scheduler). A
resumed run must pass the SAME --total-steps and --warmup-ratio as the
original run, or the reconstructed schedule won't match what it would
have been had training never been interrupted.
"""
import argparse
import dataclasses
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.model.losses import magnet_loss, script_target_rates
from src.model.magnet import MAGNET
from src.training.collate import collate_batch
from src.training.dataset import SCRIPTS, VOCAB_SIZE, MagnetByteDataset


@dataclasses.dataclass
class TrainConfig:
    # Data
    data_root: str = ""
    # First run: sw/zu/am gives a spread across script (Latin + Geez) and
    # morphology (Bantu, Bantu, Semitic) while avoiding the largest (ha)
    # and smallest (ny) corpora. Expand to all 9 by passing --languages.
    languages: tuple = ("sw", "zu", "am")
    max_seq_len: int = 512

    # Model (see [[src.model.magnet.MAGNET]] for architecture)
    d_model: int = 256
    n_heads: int = 4
    n_layers_tokenization: int = 2
    n_layers_middle: int = 6
    n_layers_final: int = 2

    # Optimization — paper Appendix D.1 starting point; tune as needed
    batch_size: int = 16
    learning_rate: float = 5e-5
    adam_betas: tuple = (0.9, 0.98)
    adam_eps: float = 1e-6
    warmup_ratio: float = 0.1
    total_steps: int = 100_000
    grad_clip_norm: float = 1.0

    # Binomial regularizer target compression rate beta, per script,
    # index-aligned with SCRIPTS = ("Latin", "Geez"). Placeholder values —
    # tune to the paper's actual per-script targets.
    beta_by_script: tuple = (0.5, 0.5)
    reg_weight: float = 1.0

    # Logging / eval / checkpointing
    log_every: int = 50
    eval_every: int = 1000
    eval_batches: int = 10
    checkpoint_every: int = 1000
    # Defaults to a local ./checkpoints/ for quick local smoke tests. In
    # Colab, point this at a Drive-mounted path (e.g.
    # /content/drive/MyDrive/DATA_ROOT/checkpoints/) — Colab's local disk
    # is wiped on disconnect, so anything not saved to Drive is lost.
    checkpoint_dir: str = "./checkpoints"
    # Must match the original run's --total-steps / --warmup-ratio when
    # resuming — see module docstring; only the step counter is restored,
    # not the LR schedule shape.
    resume_from: str = None

    device: str = None  # resolved at runtime if None


def resolve_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    print(
        "WARNING: CUDA not available, falling back to CPU. Training will be "
        "slow — real training runs are expected to happen on GPU (e.g. Colab)."
    )
    return torch.device("cpu")


class RoundRobinLanguageLoader:
    """
    Cycles through one DataLoader per language, yielding a single
    language-homogeneous batch per language in turn, so loss can be
    attributed to one language/script per step (not just a batch-mixed
    average) for [[train]]'s per-language logging.
    """

    def __init__(self, data_root, languages, split, batch_size, max_seq_len, shuffle):
        self.languages = list(languages)
        self.loaders = {
            lang: DataLoader(
                MagnetByteDataset(data_root, [lang], split=split, max_seq_len=max_seq_len),
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=collate_batch,
                drop_last=shuffle,
            )
            for lang in self.languages
        }
        self.iters = {lang: iter(loader) for lang, loader in self.loaders.items()}

    def next_batch(self, lang):
        try:
            return next(self.iters[lang])
        except StopIteration:
            self.iters[lang] = iter(self.loaders[lang])
            return next(self.iters[lang])

    def __iter__(self):
        while True:
            for lang in self.languages:
                yield lang, self.next_batch(lang)


def build_model(config):
    return MAGNET(
        vocab_size=VOCAB_SIZE,
        scripts=SCRIPTS,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers_tokenization=config.n_layers_tokenization,
        n_layers_middle=config.n_layers_middle,
        n_layers_final=config.n_layers_final,
    )


def build_optimizer(model, config):
    return torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.adam_betas,
        eps=config.adam_eps,
    )


def build_scheduler(optimizer, config):
    warmup_steps = max(1, int(config.total_steps * config.warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, config.total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_step(model, batch, config, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    script_ids = batch["script_ids"].to(device)

    logits, boundary_probs = model(input_ids, script_ids, attention_mask)
    beta = script_target_rates(script_ids, config.beta_by_script)
    loss, comps = magnet_loss(
        logits, input_ids, boundary_probs, beta,
        reg_weight=config.reg_weight, attention_mask=attention_mask,
    )
    return loss, comps


@torch.no_grad()
def evaluate(model, data_root, languages, config, device):
    model.eval()
    per_lang = {}
    for lang in languages:
        dataset = MagnetByteDataset(data_root, [lang], split="eval", max_seq_len=config.max_seq_len)
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch)

        lm_total, reg_total, n_batches = 0.0, 0.0, 0
        for batch in loader:
            if n_batches >= config.eval_batches:
                break
            _, comps = run_step(model, batch, config, device)
            lm_total += comps["lm_loss"].item()
            reg_total += comps["reg_loss"].item()
            n_batches += 1

        if n_batches > 0:
            per_lang[lang] = {"lm_loss": lm_total / n_batches, "reg_loss": reg_total / n_batches}
    model.train()
    return per_lang


def save_checkpoint(checkpoint_dir, step, model, optimizer, scheduler, config):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": dataclasses.asdict(config),
    }
    step_path = checkpoint_dir / f"step_{step}.pt"
    torch.save(payload, step_path)
    torch.save(payload, checkpoint_dir / "latest.pt")
    return step_path


def load_checkpoint(path, model, optimizer, scheduler, device):
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    return payload["step"]


def train(config):
    device = resolve_device(config.device)
    print(f"device: {device}")
    print(f"languages: {list(config.languages)}")

    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    start_step = 0
    if config.resume_from:
        start_step = load_checkpoint(config.resume_from, model, optimizer, scheduler, device) + 1
        print(f"resumed from {config.resume_from} at step {start_step}")

    train_loader = RoundRobinLanguageLoader(
        config.data_root, config.languages, split="train",
        batch_size=config.batch_size, max_seq_len=config.max_seq_len, shuffle=True,
    )

    model.train()
    t0 = time.time()
    last_completed_step = start_step - 1
    for step, (lang, batch) in enumerate(train_loader, start=start_step):
        if step >= config.total_steps:
            break

        loss, comps = run_step(model, batch, config, device)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        if step % config.log_every == 0:
            elapsed = time.time() - t0
            print(
                f"step {step} lang={lang} lr={scheduler.get_last_lr()[0]:.2e} "
                f"lm_loss={comps['lm_loss'].item():.4f} reg_loss={comps['reg_loss'].item():.4f} "
                f"elapsed={elapsed:.1f}s"
            )

        if config.eval_every and step % config.eval_every == 0 and step > start_step:
            eval_results = evaluate(model, config.data_root, config.languages, config, device)
            for eval_lang, losses in eval_results.items():
                print(
                    f"  eval step {step} lang={eval_lang} "
                    f"lm_loss={losses['lm_loss']:.4f} reg_loss={losses['reg_loss']:.4f}"
                )

        if config.checkpoint_every and step % config.checkpoint_every == 0 and step > start_step:
            path = save_checkpoint(config.checkpoint_dir, step, model, optimizer, scheduler, config)
            print(f"  checkpoint saved: {path}")

        last_completed_step = step

    final_path = save_checkpoint(config.checkpoint_dir, last_completed_step, model, optimizer, scheduler, config)
    print(f"training loop finished at step {last_completed_step}, final checkpoint: {final_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = TrainConfig()

    parser.add_argument("--data-root", type=str, default=os.environ.get("MAGNET_DATA_ROOT", defaults.data_root))
    parser.add_argument("--languages", type=str, default=",".join(defaults.languages))
    parser.add_argument("--max-seq-len", type=int, default=defaults.max_seq_len)

    parser.add_argument("--d-model", type=int, default=defaults.d_model)
    parser.add_argument("--n-heads", type=int, default=defaults.n_heads)
    parser.add_argument("--n-layers-tokenization", type=int, default=defaults.n_layers_tokenization)
    parser.add_argument("--n-layers-middle", type=int, default=defaults.n_layers_middle)
    parser.add_argument("--n-layers-final", type=int, default=defaults.n_layers_final)

    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--warmup-ratio", type=float, default=defaults.warmup_ratio)
    parser.add_argument("--total-steps", type=int, default=defaults.total_steps)
    parser.add_argument("--grad-clip-norm", type=float, default=defaults.grad_clip_norm)
    parser.add_argument("--reg-weight", type=float, default=defaults.reg_weight)
    # Comma-separated floats, index-aligned with SCRIPTS = ("Latin", "Geez"),
    # e.g. --beta-by-script 0.5,0.3. Without this flag every script shares
    # the same TrainConfig default target rate — see src/eval/compute_beta.py
    # for computing a script's target from its paper Eq. 4 byte-to-word ratio.
    parser.add_argument("--beta-by-script", type=str, default=",".join(str(b) for b in defaults.beta_by_script))

    parser.add_argument("--log-every", type=int, default=defaults.log_every)
    parser.add_argument("--eval-every", type=int, default=defaults.eval_every)
    parser.add_argument("--eval-batches", type=int, default=defaults.eval_batches)
    parser.add_argument("--checkpoint-every", type=int, default=defaults.checkpoint_every)
    # Local default is ./checkpoints/. On Colab, pass a Drive-mounted path
    # (e.g. /content/drive/MyDrive/DATA_ROOT/checkpoints/) — Colab's local
    # disk is wiped on disconnect, so checkpoints not saved to Drive are lost.
    parser.add_argument("--checkpoint-dir", type=str, default=defaults.checkpoint_dir)
    # Must use the SAME --total-steps and --warmup-ratio as the run being
    # resumed, or the LR schedule won't match — only the step counter is
    # restored from the checkpoint, not the schedule shape (see module
    # docstring and build_scheduler).
    parser.add_argument("--resume", type=str, default=defaults.resume_from)
    parser.add_argument("--device", type=str, default=defaults.device)

    args = parser.parse_args()
    return TrainConfig(
        data_root=args.data_root,
        languages=tuple(args.languages.split(",")),
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers_tokenization=args.n_layers_tokenization,
        n_layers_middle=args.n_layers_middle,
        n_layers_final=args.n_layers_final,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        total_steps=args.total_steps,
        grad_clip_norm=args.grad_clip_norm,
        reg_weight=args.reg_weight,
        beta_by_script=tuple(float(b) for b in args.beta_by_script.split(",")),
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume,
        device=args.device,
    )


def main():
    config = parse_args()
    if not config.data_root:
        raise SystemExit("Set --data-root or the MAGNET_DATA_ROOT environment variable.")
    train(config)


if __name__ == "__main__":
    main()
