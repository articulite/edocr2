"""Fine-tune a recognizer on drawgen crops (training/collect_crops.py) and synthetic strips
(training/synth.py), on the GPU.

    python training/train.py --kind dim --crops DATA/crops/shard0 [DATA/crops/shard1 ...] \
        --init edocr2/models/recognizer_dimensions_2.pt --out NEW.pt [--epochs 8]

Starts from the released weights. For the dimension recognizer the alphabet grows
(training/alphabets.py): the output layer is rebuilt with every old character's weights in
place, so only the new characters start from nothing. Batch-norm statistics stay as trained
(fine-tuning on a narrower set of drawings should not move them). One crop in ten, by file, is
held out to report accuracy; the real test is the benchmarks.
"""

import argparse, json, os, random, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from edocr2.torch_recognizer import CRNN, TorchRecognizer    # noqa: E402
from training.alphabets import DIMENSIONS, GDT                 # noqa: E402
from training.synth import Synth                               # noqa: E402


def widen(old, new_alphabet):
    """A CRNN for new_alphabet with old's weights; the output rows of old characters (and the
    blank) moved to their new places."""
    net = CRNN(len(new_alphabet) + 1)
    state = {k: v for k, v in old.net.state_dict().items() if not k.startswith("fc12.")}
    net.load_state_dict(state, strict=False)
    with torch.no_grad():
        w_old, b_old = old.net.fc12.weight, old.net.fc12.bias
        net.fc12.weight.normal_(0, 0.01)
        net.fc12.bias.fill_(float(b_old.min()))
        for i, ch in enumerate(old.alphabet):
            j = new_alphabet.index(ch)
            net.fc12.weight[j], net.fc12.bias[j] = w_old[i], b_old[i]
        net.fc12.weight[-1], net.fc12.bias[-1] = w_old[-1], b_old[-1]   # the blank
    return net


def load_crops(folders, alphabet):
    items = []
    for f in folders:
        for kind_dir in [Path(f)]:
            for line in open(kind_dir / "labels.jsonl", encoding="utf-8"):
                r = json.loads(line)
                if r["label"] and all(ch in alphabet for ch in r["label"]):
                    items.append((str(kind_dir / r["file"]), r["label"]))
    return items


def roughen(img, rng):
    """Light changes a crop might come with: stroke weight, blur, a few pixels of margin."""
    if rng.random() < 0.3:
        k = np.ones((2, 2), np.uint8)
        img = cv2.erode(img, k) if rng.random() < 0.5 else cv2.dilate(img, k)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.3, 0.9))
    if rng.random() < 0.3:
        p = rng.randint(1, 6)
        img = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("dim", "gdt"), required=True)
    ap.add_argument("--crops", nargs="+", required=True, help="collect_crops output folders (each holding dim/ and gdt/)")
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--synth", type=float, default=0.5, help="synthetic strips per real crop")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    alphabet = DIMENSIONS if args.kind == "dim" else GDT
    base = TorchRecognizer.load(args.init, device="cpu")
    net = widen(base, alphabet) if base.alphabet != alphabet else base.net
    rec = TorchRecognizer(alphabet, device=device)
    rec.net = net.to(device)
    blank = len(alphabet)

    items = load_crops([os.path.join(c, args.kind) for c in args.crops], alphabet)
    rng.shuffle(items)
    held = items[: len(items) // 10]
    train = items[len(items) // 10:]
    synth = Synth(args.kind, alphabet, seed=args.seed)
    print(f"{len(train)} training crops, {len(held)} held out, alphabet {len(alphabet)}", flush=True)

    def batch(pairs):
        x = torch.from_numpy(np.stack([rec.prepare(img) for img, _ in pairs])).to(device)
        targets = torch.tensor([alphabet.index(ch) for _, t in pairs for ch in t], dtype=torch.long)
        lengths = torch.tensor([len(t) for _, t in pairs], dtype=torch.long)
        return x, targets, lengths

    def accuracy(pairs):
        rec.net.eval()
        got = []
        for i in range(0, len(pairs), 256):
            got += rec.recognize_batch([img for img, _ in pairs[i:i + 256]])
        return sum(g == t for g, (_, t) in zip(got, pairs)) / max(1, len(pairs))

    held_pairs = [(cv2.imread(p), t) for p, t in held]
    synth_check = [synth() for _ in range(500)]
    print(f"before: held-out crops {accuracy(held_pairs):.3f}, synthetic {accuracy(synth_check):.3f}", flush=True)

    opt = torch.optim.Adam(rec.net.parameters(), lr=args.lr)
    n_synth = int(args.batch * args.synth / (1 + args.synth))
    per_step = args.batch - n_synth
    steps_per_epoch = (len(train) + per_step - 1) // per_step
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.15)
    for epoch in range(args.epochs):
        rng.shuffle(train)
        rec.net.train()
        for m in rec.net.modules():          # batch-norm statistics stay as trained
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
        started, total = time.time(), 0.0
        for s in range(steps_per_epoch):
            real = [(roughen(cv2.imread(p), rng), t) for p, t in train[s * per_step:(s + 1) * per_step]]
            pairs = real + [synth() for _ in range(n_synth)]
            x, targets, lengths = batch(pairs)
            logp = F.log_softmax(rec.net.logits(x), dim=2).permute(1, 0, 2)          # (T, B, C)
            frames = torch.full((len(pairs),), logp.shape[0], dtype=torch.long)
            loss = F.ctc_loss(logp, targets, frames, lengths, blank=blank, zero_infinity=True)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rec.net.parameters(), 5.0)
            opt.step()
            sched.step()
            total += float(loss)
        print(f"epoch {epoch + 1}: loss {total / steps_per_epoch:.3f}, held-out crops {accuracy(held_pairs):.3f}, "
              f"synthetic {accuracy(synth_check):.3f}, {time.time() - started:.0f} s", flush=True)
    rec.net.eval()
    rec.save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
