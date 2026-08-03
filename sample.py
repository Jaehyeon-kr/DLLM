import os
import argparse
import torch
from transformers import AutoModelForMaskedLM

from tokenizer import get_tokenizer

### rich is optional. inference.py depends on it, but sample.py should still run on a bare ###
### environment (e.g. the local box) where it isn't installed, so we fall back to plain print. ###
try:
    from rich.live import Live
    from rich.console import Console
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.text import Text
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


def parse_args():
    parser = argparse.ArgumentParser(description="Masked Diffusion LM Sampling")

    parser.add_argument(
        "--hf_model_name",
        help="Must match what was used in pretrain.py so the vocab lines up",
        default="answerdotai/ModernBERT-base",
        type=str
    )

    parser.add_argument(
        "--path_to_checkpoint",
        help="Checkpoint directory written by accelerator.save_state. If omitted, the raw \
            huggingface weights are used, which have NOT seen this objective",
        default=None,
        type=str
    )

    parser.add_argument(
        "--prompt",
        help="Text to condition on. Held fixed at every step",
        default="",
        type=str
    )

    parser.add_argument(
        "--gen_length",
        help="How many tokens to generate. Diffusion fills a fixed canvas, so this is decided up front",
        default=128,
        type=int
    )

    parser.add_argument(
        "--num_steps",
        help="Denoising steps. num_steps == gen_length means roughly one token per step",
        default=128,
        type=int
    )

    parser.add_argument(
        "--block_length",
        help="Semi-autoregressive block size. 0 disables blocking and denoises the whole \
            canvas at once",
        default=32,
        type=int
    )

    parser.add_argument(
        "--strategy",
        help="ancestral matches the training objective exactly. confidence is biased but \
            much stronger at low step counts",
        default="confidence",
        choices=["ancestral", "confidence"],
        type=str
    )

    parser.add_argument(
        "--temperature",
        default=1.0,
        type=float
    )

    parser.add_argument(
        "--top_p",
        default=0.95,
        type=float
    )

    parser.add_argument(
        "--num_samples",
        default=1,
        type=int
    )

    parser.add_argument(
        "--seed",
        default=None,
        type=int
    )

    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        type=str
    )

    parser.add_argument(
        "--show_steps",
        help="Print the partially filled canvas after every step",
        default=False,
        action=argparse.BooleanOptionalAction
    )

    return parser.parse_args()


def load_checkpoint(model, path_to_checkpoint):
    """accelerator.save_state writes optimizer/scheduler state alongside the model, so this
    is not a from_pretrained directory. Pull the weights out by hand."""

    candidates = ["model.safetensors", "pytorch_model.bin"]

    for name in candidates:
        path = os.path.join(path_to_checkpoint, name)
        if not os.path.exists(path):
            continue

        if name.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(path)
        else:
            state_dict = torch.load(path, map_location="cpu")

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded {path}")
        if missing:
            print(f"  missing keys: {len(missing)} (first few: {missing[:5]})")
        if unexpected:
            print(f"  unexpected keys: {len(unexpected)} (first few: {unexpected[:5]})")
        return model

    raise FileNotFoundError(
        f"No {' or '.join(candidates)} in {path_to_checkpoint}. Contents: {os.listdir(path_to_checkpoint)}"
    )


def sample_from_logits(logits, temperature, top_p, forbidden_token_id=None):
    """logits: (B, L, V) -> (token_ids, confidence), both (B, L).

    Confidence is the probability the model assigns to the token we drew, which is what the
    confidence strategy ranks on."""

    if forbidden_token_id is not None:

        ### [MASK] is a real vocab entry the model can emit. If it ever wins, that position ###
        ### still looks unfilled on the next step and the bookkeeping below drifts ###
        logits = logits.clone()
        logits[..., forbidden_token_id] = float("-inf")

    if temperature > 0:
        logits = logits / temperature

    probs = torch.softmax(logits.float(), dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)

        ### Keep everything up to and including the token that crosses top_p ###
        keep = (cumulative - sorted_probs) < top_p
        sorted_probs = sorted_probs * keep
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        ### Scatter back so probs stays aligned with the vocab ###
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)

    batch_size, seq_len, vocab_size = probs.shape
    flat = probs.reshape(-1, vocab_size)

    if temperature > 0:
        drawn = torch.multinomial(flat, num_samples=1)
    else:
        drawn = flat.argmax(dim=-1, keepdim=True)

    confidence = flat.gather(-1, drawn)

    return drawn.reshape(batch_size, seq_len), confidence.reshape(batch_size, seq_len)


def render_canvas(tokenizer, sequence, mask_token_id):
    """One line showing which positions are still masked. Decoding the whole row keeps the
    subwords glued together, which piecewise decoding would not."""

    text = tokenizer.decode(sequence.tolist(), skip_special_tokens=False)

    ### [MASK] spelled out is most of the line early on, so shrink it to a single character. ###
    ### ASCII only, since a cp949 console raises on anything fancier ###
    text = text.replace(tokenizer.mask_token, "_")

    ### Newlines in the generated text would break the one line per step layout ###
    return text.replace("\n", "\\n")


def render_canvas_rich(tokenizer, sequence, mask_token_id):
    """rich.Text version of render_canvas: still-masked positions are dimmed to '_', decided
    tokens are shown normally. Decoding per token loses the subword gluing render_canvas gets
    from decoding the whole row, but token level color is the whole point here."""

    out = Text()
    ids = sequence.tolist()
    for tok_id in ids:
        if tok_id == mask_token_id:
            out.append("_", style="dim")
        else:
            piece = tokenizer.decode([tok_id], skip_special_tokens=False)
            piece = piece.replace("\n", "\\n")
            out.append(piece, style="white")
    return out


@torch.inference_mode()
def denoise_span(model, tokenizer, x, editable, mask_token_id, num_steps, args,
                 live=None, progress=None, task=None):
    """Run the reverse process over the positions marked editable. Everything else, prompt
    and already finished blocks alike, is held fixed by construction.

    live/progress/task are the shared rich widgets created in generate(). When they are None
    (rich missing, or show_steps off) we fall back to the plain print path."""

    attention_mask = torch.ones_like(x)
    num_editable = int(editable[0].sum().item())

    for step in range(num_steps):

        ### t counts down from 1 (all masked) to 0 (all decided) ###
        t_now = 1.0 - step / num_steps
        t_next = 1.0 - (step + 1) / num_steps

        masked = editable & (x == mask_token_id)
        if not masked.any():
            break

        logits = model(input_ids=x, attention_mask=attention_mask)["logits"]
        predicted, confidence = sample_from_logits(logits, args.temperature, args.top_p,
                                                   forbidden_token_id=mask_token_id)

        if args.strategy == "ancestral":

            ### Each masked position independently commits with probability (t_now - t_next) / t_now. ###
            ### Once committed it is never remasked, the absorbing state the training objective assumes ###
            p_commit = (t_now - t_next) / t_now
            commit = masked & (torch.rand_like(x, dtype=torch.float32) < p_commit)

        else:

            ### Commit the highest confidence tokens first and let the schedule decide how many. ###
            ### Biased relative to the objective, but far stronger when num_steps << gen_length ###
            target_committed = round(num_editable * (1.0 - t_next))
            already_committed = num_editable - int(masked[0].sum().item())
            num_to_commit = max(1, target_committed - already_committed)
            num_to_commit = min(num_to_commit, int(masked[0].sum().item()))

            ranked = confidence.masked_fill(~masked, float("-inf"))
            chosen = ranked.topk(num_to_commit, dim=-1).indices

            commit = torch.zeros_like(masked).scatter_(1, chosen, True)

        ### topk can reach past the masked positions when rows fall out of lockstep, so pin ###
        ### the write to the editable region no matter which branch produced it ###
        commit = commit & masked

        x = torch.where(commit, predicted, x)

        if args.show_steps:
            num_masked = int((x == mask_token_id).sum().item())

            ### rich path: update the shared Live canvas + progress bar in place. Only the ###
            ### first row, since the rows fall out of lockstep and printing all of them ###
            ### buries the one you are actually watching. ###
            if live is not None:
                canvas = render_canvas_rich(tokenizer, x[0], mask_token_id)
                header = Text(f"t={t_next:.3f}  masked={num_masked}\n", style="bold green")
                live.update(header + canvas)
                if progress is not None and task is not None:
                    progress.advance(task, 1)
            else:
                print(f"  [step {step + 1}/{num_steps}] t={t_next:.3f} masked={num_masked}")
                print(f"    {render_canvas(tokenizer, x[0], mask_token_id)}")

    ### The schedule should have committed everything by t=0, but never hand a half masked ###
    ### block to the next one. Fill any stragglers greedily ###
    leftover = editable & (x == mask_token_id)
    if leftover.any():
        logits = model(input_ids=x, attention_mask=attention_mask)["logits"]
        predicted, _ = sample_from_logits(logits, args.temperature, args.top_p,
                                          forbidden_token_id=mask_token_id)
        x = torch.where(leftover, predicted, x)

    return x


@torch.inference_mode()
def generate(model, tokenizer, args, device):

    mask_token_id = tokenizer.mask_token_id

    ### Encode the prompt without the trailing EOS the template would add, since the model ###
    ### is meant to keep writing past it ###
    prompt_ids = tokenizer(args.prompt, add_special_tokens=False)["input_ids"]
    prompt_ids = [tokenizer.bos_token_id] + prompt_ids
    prompt_length = len(prompt_ids)

    total_length = prompt_length + args.gen_length

    x = torch.full((args.num_samples, total_length), mask_token_id, dtype=torch.long, device=device)
    x[:, :prompt_length] = torch.tensor(prompt_ids, dtype=torch.long, device=device)

    ### Blocking left to right suits a model trained on packed documents better than denoising ###
    ### the whole canvas at once, since it never learned to lay out a standalone length L document ###
    block_length = args.block_length if args.block_length > 0 else args.gen_length
    num_blocks = (args.gen_length + block_length - 1) // block_length
    steps_per_block = max(1, args.num_steps // num_blocks)

    def run_blocks(live=None, progress=None, task=None):
        nonlocal x
        for block in range(num_blocks):
            start = prompt_length + block * block_length
            end = min(start + block_length, total_length)

            editable = torch.zeros_like(x, dtype=torch.bool)
            editable[:, start:end] = True

            if args.show_steps and live is None:
                print(f"[block {block + 1}/{num_blocks}] positions {start}:{end}")

            x = denoise_span(model, tokenizer, x, editable, mask_token_id, steps_per_block, args,
                             live=live, progress=progress, task=task)

    ### With rich available and show_steps on, drive the whole run through one Live canvas + ###
    ### progress bar. steps_per_block * num_blocks is the true total the bar counts up to. ###
    if args.show_steps and _HAS_RICH:
        console = Console(highlight=False)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Denoising...", total=steps_per_block * num_blocks)
            with Live("", refresh_per_second=8, console=console) as live:
                run_blocks(live=live, progress=progress, task=task)
    else:
        run_blocks()

    return x, prompt_length


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)

    tokenizer = get_tokenizer(args.hf_model_name)

    model = AutoModelForMaskedLM.from_pretrained(args.hf_model_name)
    model.resize_token_embeddings(len(tokenizer))

    if args.path_to_checkpoint is not None:
        load_checkpoint(model, args.path_to_checkpoint)
    else:
        print("WARNING: no checkpoint given, sampling from the raw huggingface weights")

    model = model.to(device=device, dtype=dtype)
    model.eval()

    x, prompt_length = generate(model, tokenizer, args, device)

    for i, sequence in enumerate(x):
        generated = sequence[prompt_length:].tolist()

        ### The model saw EOS delimited documents during packing, so treat the first one as a stop ###
        if tokenizer.eos_token_id in generated:
            generated = generated[:generated.index(tokenizer.eos_token_id)]

        print(f"\n===== sample {i + 1}/{args.num_samples} =====")
        if args.prompt:
            print(f"[prompt] {args.prompt}")
        print(tokenizer.decode(generated, skip_special_tokens=False))


if __name__ == "__main__":
    main()
