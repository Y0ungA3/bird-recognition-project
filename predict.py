import argparse, json, os
from pathlib import Path
root=Path(__file__).resolve().parent
os.environ.setdefault('TORCH_HOME',str(root/'cache'/'torch'))
import torch
from PIL import Image
from torchvision import models, transforms

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('image',type=Path)
    p.add_argument('--checkpoint',type=Path,default=root/'runs'/'baseline_full'/'best.pt')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if ckpt['config']['crop']:
        raise ValueError('This checkpoint was trained on ground-truth bird crops. Supply a manually cropped bird image using a separate crop-aware workflow; use the full-image baseline for this predictor.')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=models.resnet18(weights=None); model.fc=torch.nn.Linear(512,200)
    model.load_state_dict(ckpt['model']); model.to(device).eval()
    tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    with Image.open(args.image) as im: x=tf(im.convert('RGB')).unsqueeze(0).to(device)
    with torch.inference_mode(): scores,indices=model(x).softmax(1)[0].topk(5)
    result=[{'class':ckpt['classes'][i],'score':float(v)} for v,i in zip(scores.cpu().tolist(),indices.cpu().tolist())]
    output=json.dumps(result,ensure_ascii=False,indent=2)
    print(output)
    if args.output: args.output.write_text(output,encoding='utf-8')
