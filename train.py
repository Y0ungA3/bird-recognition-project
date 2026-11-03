import argparse, csv, hashlib, json, os, platform, random, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('TORCH_HOME', str(ROOT / 'cache' / 'torch'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'cache' / 'matplotlib'))
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


def read_pairs(path):
    return dict(line.strip().split(maxsplit=1) for line in path.read_text().splitlines())


def make_splits(data, seed):
    images = read_pairs(data / 'images.txt')
    labels = read_pairs(data / 'image_class_labels.txt')
    official = read_pairs(data / 'train_test_split.txt')
    boxes = {line.split()[0]: list(map(float, line.split()[1:])) for line in (data / 'bounding_boxes.txt').read_text().splitlines()}
    groups, test = {}, []
    for key, name in images.items():
        row = {'id': int(key), 'path': name, 'label': int(labels[key]) - 1, 'bbox': boxes[key]}
        if official[key] == '1':
            groups.setdefault(row['label'], []).append(row)
        else:
            test.append(row)
    train, val = [], []
    rng = random.Random(seed)
    for label in sorted(groups):
        rows = sorted(groups[label], key=lambda x: x['id'])
        rng.shuffle(rows)
        n = max(1, round(len(rows) * .2))
        val.extend(rows[:n]); train.extend(rows[n:])
    assert not (set(x['id'] for x in train) & set(x['id'] for x in val))
    assert len(train) + len(val) + len(test) == 11788
    return {'train': train, 'val': val, 'test': test}


class Letterbox:
    """Resize the entire RGB input into a square without removing any region."""
    def __init__(self, size=224, fill=(124,116,104)):
        self.size, self.fill = size, fill
    def __call__(self, image):
        scale = self.size / max(image.size)
        width = min(self.size, max(1, round(image.width * scale)))
        height = min(self.size, max(1, round(image.height * scale)))
        resized = image.resize((width, height), Image.Resampling.BILINEAR)
        canvas = Image.new('RGB', (self.size, self.size), self.fill)
        canvas.paste(resized, ((self.size-width)//2, (self.size-height)//2))
        return canvas


class Birds(Dataset):
    def __init__(self, data, rows, training, crop, margin, resize_mode="legacy"):
        self.data, self.rows, self.crop, self.margin = data, rows, crop, margin
        geometry = ([Letterbox(224)] + ([transforms.RandomHorizontalFlip()] if training else [])) if resize_mode == 'letterbox' else ([transforms.RandomResizedCrop(224, scale=(.7, 1.)), transforms.RandomHorizontalFlip()] if training else [transforms.Resize(256), transforms.CenterCrop(224)])
        self.tf = transforms.Compose([
            *geometry,
            transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.data / 'images' / row['path']) as im:
            im = im.convert('RGB')
            if self.crop:
                x, y, w, h = row['bbox']
                im = im.crop((max(0, x-w*self.margin), max(0, y-h*self.margin), min(im.width, x+w*(1+self.margin)), min(im.height, y+h*(1+self.margin))))
            return self.tf(im), row['label'], row['id']


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed); np.random.seed(seed)


def metrics(cm):
    tp = cm.diag().float(); actual = cm.sum(1).float(); predicted = cm.sum(0).float()
    f1 = 2*tp / (actual + predicted).clamp_min(1)
    return {'accuracy': (tp.sum()/cm.sum().clamp_min(1)).item(), 'macro_f1': f1.mean().item()}


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None, frozen=False):
    training = optimizer is not None
    model.train(training)
    if frozen:
        model.eval(); model.fc.train()
    total_loss = count = top5 = 0
    cm = torch.zeros(200, 200, dtype=torch.int64)
    predictions = []
    for step, (images, labels, ids) in enumerate(loader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(device_type='cuda', enabled=device.type == 'cuda'):
            logits = model(images)
            loss = criterion(logits, labels)
        if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
        if training:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        guesses = logits.detach().topk(5, dim=1).indices
        top5 += guesses.eq(labels[:, None]).any(1).sum().item()
        pred, target = guesses[:, 0].cpu(), labels.cpu()
        cm += torch.bincount(target*200 + pred, minlength=40000).reshape(200, 200)
        count += len(labels); total_loss += loss.item()*len(labels)
        predictions.extend((int(i), int(y), int(p)) for i,y,p in zip(ids, target, pred))
        if training and (step + 1) % 25 == 0:
            print(f'  batch {step+1}/{len(loader)} loss={total_loss/count:.4f}', flush=True)
    return {'loss': total_loss/count, **metrics(cm), 'top5': top5/count}, cm, predictions


def plot_history(path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader((path/'history.csv').open()))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for split in ['train', 'val']:
        axes[0].plot([int(r['epoch']) for r in rows], [float(r[f'{split}_loss']) for r in rows], label=split)
        axes[1].plot([int(r['epoch']) for r in rows], [float(r[f'{split}_accuracy']) for r in rows], label=split)
    for ax, title in zip(axes, ['Cross entropy', 'Top-1 accuracy']):
        ax.set(xlabel='Epoch', title=title); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(path/'learning_curves.png', dpi=160); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=ROOT/'data'/'CUB_200_2011')
    p.add_argument('--run', type=Path, default=ROOT/'runs'/'baseline_full')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--freeze-epochs', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--split-seed', type=int, default=42, help='Keep the same train/validation split across training seeds')
    p.add_argument('--crop', action='store_true')
    p.add_argument('--resize-mode', choices=['legacy','letterbox'], default='legacy')
    p.add_argument('--margin', type=float, default=.1)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--evaluate', action='store_true', help='Explicit final evaluation on official test set')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--smoke', action='store_true', help='Small pipeline check; not a reportable benchmark')
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU unavailable; refusing silent CPU fallback')
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    args.run.mkdir(parents=True, exist_ok=True)
    saved = None
    if args.evaluate or args.resume:
        saved = torch.load(args.checkpoint or args.run/('best.pt' if args.evaluate else 'last.pt'), map_location='cpu', weights_only=False)
        for key in ['seed', 'crop', 'margin', 'freeze_epochs', 'smoke']:
            setattr(args, key, saved['config'][key])
        args.split_seed = saved['config'].get('split_seed', saved['config']['seed'])
        args.resize_mode = saved['config'].get('resize_mode', 'legacy')
    elif (args.run/'config.json').exists():
        raise RuntimeError('Run directory already used. Choose another --run or use --resume.')
    if args.resize_mode == 'letterbox' and not args.crop:
        raise ValueError('Letterbox experiment requires --crop')
    splits = make_splits(args.data, args.split_seed)
    split_hash = hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()
    if saved and saved['split_sha256'] != split_hash:
        raise RuntimeError('Dataset split differs from checkpoint')
    classes = list(read_pairs(args.data/'classes.txt').values())
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    info = {'python': platform.python_version(), 'torch': torch.__version__, 'torchvision': __import__('torchvision').__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(), 'split_sha256': split_hash, 'sizes': {k:len(v) for k,v in splits.items()}}
    print(json.dumps(info, indent=2), flush=True)
    if not args.evaluate:
        (args.run/'config.json').write_text(json.dumps(config, indent=2))
        (args.run/'environment.json').write_text(json.dumps(info, indent=2))
        (args.run/'splits.json').write_text(json.dumps({k:[r['id'] for r in v] for k,v in splits.items()}))
    def loader(split):
        rows = splits[split][:128 if split=='train' else 64] if args.smoke else splits[split]
        gen = torch.Generator().manual_seed(args.seed)
        return DataLoader(Birds(args.data, rows, split=='train', args.crop, args.margin, args.resize_mode), batch_size=args.batch_size, shuffle=split=='train', num_workers=args.workers, pin_memory=True, persistent_workers=args.workers>0, worker_init_fn=seed_worker, generator=gen)
    model = models.resnet18(weights=None if saved else models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 200)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    if saved: model.load_state_dict(saved['model'])
    if args.evaluate:
        model.eval()
        result, cm, preds = run_epoch(model, loader('test'), device, criterion)
        result.update({'checkpoint_epoch':saved['epoch'], 'split_sha256':split_hash, 'smoke':args.smoke})
        (args.run/'test_metrics.json').write_text(json.dumps(result, indent=2))
        np.savetxt(args.run/'test_confusion_matrix.csv', cm.numpy(), delimiter=',', fmt='%d')
        with (args.run/'test_predictions.csv').open('w', newline='') as f:
            writer=csv.writer(f); writer.writerow(['image_id','true_class_0based','pred_class_0based']); writer.writerows(preds)
        print('TEST:', result, flush=True); return
    optimizer = torch.optim.AdamW([{'params': [v for k,v in model.named_parameters() if not k.startswith('fc.')], 'lr':1e-4}, {'params':model.fc.parameters(), 'lr':1e-3}], weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    start, best = 0, -1.
    if args.resume:
        optimizer.load_state_dict(saved['optimizer']); scaler.load_state_dict(saved['scaler'])
        start, best = saved['epoch'], saved['best_val_accuracy']
    train_loader, val_loader = loader('train'), loader('val')
    history = args.run/'history.csv'
    for epoch in range(start, args.epochs):
        frozen = epoch < args.freeze_epochs
        for name, param in model.named_parameters(): param.requires_grad = name.startswith('fc.') or not frozen
        begin = time.time()
        tr, _, _ = run_epoch(model, train_loader, device, criterion, optimizer, scaler, frozen)
        va, _, _ = run_epoch(model, val_loader, device, criterion)
        row = {'epoch':epoch+1,'phase':'head' if frozen else 'finetune','seconds':round(time.time()-begin,2), **{'train_'+k:v for k,v in tr.items()}, **{'val_'+k:v for k,v in va.items()}}
        new_best = va['accuracy'] > best
        best = max(best, va['accuracy'])
        state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'scaler':scaler.state_dict(), 'epoch':epoch+1, 'best_val_accuracy':best, 'config':config, 'classes':classes, 'split_sha256':split_hash}
        for name in ['last.pt'] + (['best.pt'] if new_best else []):
            temp = args.run/(name+'.tmp'); torch.save(state, temp); temp.replace(args.run/name)
        with history.open('a', newline='') as f:
            writer=csv.DictWriter(f, fieldnames=list(row))
            if f.tell()==0: writer.writeheader()
            writer.writerow(row)
        plot_history(args.run)
        print(json.dumps(row), flush=True)
    (args.run/'completed.json').write_text(json.dumps({'epochs':args.epochs, 'best_val_accuracy':best, 'smoke':args.smoke}, indent=2))
    print('TRAINING COMPLETE. Test set remains untouched; use --evaluate after choosing the final method.', flush=True)


if __name__ == '__main__':
    main()
