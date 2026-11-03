import json, random, os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent/'cache'/'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle
root=Path(__file__).resolve().parent
data=root/'data'/'CUB_200_2011'
images=dict(line.split(maxsplit=1) for line in (data/'images.txt').read_text().splitlines())
boxes={line.split()[0]: list(map(float,line.split()[1:])) for line in (data/'bounding_boxes.txt').read_text().splitlines()}
ids=random.Random(42).sample(list(images),12)
fig,axes=plt.subplots(3,4,figsize=(12,9))
for ax,key in zip(axes.flat,ids):
    with Image.open(data/'images'/images[key]) as im:
        ax.imshow(im.convert('RGB'))
    x,y,w,h=boxes[key];ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='red',linewidth=1.5))
    ax.set_title(images[key].split('/')[0][4:].replace('_',' '),fontsize=8);ax.axis('off')
fig.suptitle('CUB-200-2011: sample images and annotated bird boxes')
fig.tight_layout();fig.savefig(root/'data'/'dataset_preview.png',dpi=150)
