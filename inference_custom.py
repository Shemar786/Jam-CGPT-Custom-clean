import torch
from model import GPT
from myutils import prep, seq2sent, index2word
from types import SimpleNamespace


def load_model(ckpt_path, device):
    """
    Load the GPT model from a checkpoint, converting saved dict to namespace for model args.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    model_args = ckpt['model_args']
    # if model_args was saved as a dict, wrap into a simple namespace
    if isinstance(model_args, dict):
        model_args = SimpleNamespace(**model_args)
    # ensure LoRA config attributes exist with defaults
    for attr, default in [('lora_dropout', 0.0), ('lora_r', 0), ('lora_alpha', 1.0)]:
        if not hasattr(model_args, attr):
            setattr(model_args, attr, default)
    model = GPT(model_args)
    # allow missing buffers like attn.bias
    model.load_state_dict(ckpt['model'], strict=False)
    model.to(device).eval()
    return model


def describe_method(model, method_str, device, max_new_tokens=50):
    """
    Generate an English description for a given Java method.
    """
    # Some utils.prep implementations expect a tab-delimited line (code\tcomment)
    # We fake a blank comment after a tab so prep returns token IDs
    fake_line = method_str.strip() + "\t"
    ids = prep(fake_line)
    if ids is None:
        raise RuntimeError(f"prep returned None for input: {fake_line!r}")
    x = torch.tensor([ids], dtype=torch.long, device=device)
    y = model.generate(x, max_new_tokens=max_new_tokens, top_k=50)
    return seq2sent(y[0].tolist(), index2word)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate English descriptions for Java methods"
    )
    parser.add_argument(
        "--ckpt", default="out-jam-cgpt/ckpt_len1024.pt",
        help="Path to your fine-tuned .pt checkpoint"
    )
    parser.add_argument(
        "--method", required=True,
        help="A single Java method, wrapped in quotes"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=50,
        help="How many tokens to generate"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)

    print("\n=== Input Java Method ===\n")
    print(args.method)
    print("\n=== Generated Description ===\n")
    print(describe_method(model, args.method, device, args.max_new_tokens))
    print()
