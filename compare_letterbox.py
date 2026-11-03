"""Compare full, center crop and whole-region letterbox on the fixed validation set."""
import csv,json,os,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache'/'matplotlib'))
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import models
from train import Birds, make_splits, run_epoch, read_pairs
OUT=ROOT/'runs'/'letterbox_comparison'
MODES=['full','crop','letterbox'];COLORS=['#4477aa','#ee7733','#228833'];NAMES=['Full image','Center crop','Letterbox']

def save_csv(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    old=ROOT/'runs'/'comparison';oldsummary=json.loads((old/'summary.json').read_text())
    data=ROOT/'data'/'CUB_200_2011';rows=make_splits(data,42)['val'];labels={r['id']:r['label'] for r in rows}
    classes=list(read_pairs(data/'classes.txt').values())
    results=[]
    for r in csv.DictReader((old/'validation_results.csv').open(encoding='utf-8-sig')):
        results.append({k:int(v) if k in ['seed','best_epoch'] else v if k=='mode' else float(v) for k,v in r.items()})
    preds=[]
    for r in csv.DictReader((old/'validation_predictions.csv').open(encoding='utf-8-sig')):
        preds.append({k:v if k=='mode' else int(v) for k,v in r.items()})
    for seed in [42,43,44]:
        path=ROOT/'runs'/f'letterbox_seed{seed}'
        assert (path/'completed.json').exists()
        ck=torch.load(path/'best.pt',map_location='cpu',weights_only=False);cfg=ck['config']
        assert ck['split_sha256']==oldsummary['split_sha256']
        assert cfg['resize_mode']=='letterbox' and cfg['crop'] and cfg['seed']==seed and cfg['split_seed']==42 and not cfg['smoke']
        expected={'epochs':10,'freeze_epochs':2,'batch_size':32,'workers':4,'margin':.1}
        assert all(cfg[k]==v for k,v in expected.items())
        assert json.loads((path/'splits.json').read_text())['val']==[r['id'] for r in rows]
        model=models.resnet18(weights=None);model.fc=torch.nn.Linear(512,200);model.load_state_dict(ck['model']);model.cuda().eval()
        loader=DataLoader(Birds(data,rows,False,True,.1,'letterbox'),batch_size=32,num_workers=0,pin_memory=True)
        with torch.inference_mode():m,cm,pr=run_epoch(model,loader,torch.device('cuda'),torch.nn.CrossEntropyLoss())
        result={'seed':seed,'mode':'letterbox','best_epoch':ck['epoch'],**m};results.append(result)
        (path/'validation_metrics.json').write_text(json.dumps(result,indent=2))
        for i,y,p in pr:preds.append({'seed':seed,'mode':'letterbox','image_id':i,'true_class_0based':y,'predicted_class_0based':p,'correct':int(y==p)})
        print('Validated:',result,flush=True)
        del model,loader;torch.cuda.empty_cache()
    assert len(preds)==10800 and len({(r['seed'],r['mode'],r['image_id']) for r in preds})==10800
    for r in preds:assert r['true_class_0based']==labels[r['image_id']]
    save_csv('validation_results.csv',results);save_csv('validation_predictions.csv',preds)
    pm={(s,m):{r['image_id']:r['predicted_class_0based'] for r in preds if r['seed']==s and r['mode']==m} for s in [42,43,44] for m in MODES}
    summary={m:{k:{'mean':float(np.mean([r[k] for r in results if r['mode']==m])),'std':float(np.std([r[k] for r in results if r['mode']==m],ddof=1))} for k in ['accuracy','top5','macro_f1']} for m in MODES}
    paired=[]
    for seed in [42,43,44]:
        for reference in ['full','crop']:
            fixes=sum(pm[(seed,reference)][r['id']]!=r['label'] and pm[(seed,'letterbox')][r['id']]==r['label'] for r in rows)
            breaks=sum(pm[(seed,reference)][r['id']]==r['label'] and pm[(seed,'letterbox')][r['id']]!=r['label'] for r in rows)
            paired.append({'seed':seed,'reference':reference,'letterbox_delta_percentage_points':100*(fixes-breaks)/len(rows),'fixed_errors':fixes,'new_errors':breaks})
    save_csv('paired_differences.csv',paired)
    areas={}
    for r in rows:
        with Image.open(data/'images'/r['path']) as im:areas[r['id']]=r['bbox'][2]*r['bbox'][3]/(im.width*im.height)
    binned=[]
    for low,high,name in [(0,.1,'<10%'),(.1,.3,'10-30%'),(.3,.6,'30-60%'),(.6,float('inf'),'>=60%')]:
        ids=[r['id'] for r in rows if low<=areas[r['id']]<high]
        row={'bird_box_area_group':name,'images':len(ids)}
        for mode in MODES:
            values=[np.mean([pm[(s,mode)][i]==labels[i] for i in ids]) for s in [42,43,44]]
            row[mode+'_mean']=float(np.mean(values));row[mode+'_std']=float(np.std(values,ddof=1))
        binned.append(row)
    save_csv('bird_size_analysis.csv',binned)
    summary.update({'paired':paired,'bird_size_analysis':binned,'training_seeds':[42,43,44],'split_seed':42,'split_sha256':oldsummary['split_sha256'],'official_test_evaluated':False,'train_images':4794,'validation_images':1200,'old_results_sha256':hashlib.sha256((old/'validation_results.csv').read_bytes()).hexdigest(),'train_source_sha256':hashlib.sha256((ROOT/'train.py').read_bytes()).hexdigest(),'padding_rgb':[124,116,104],'new_training_geometry':'whole-region letterbox + random horizontal flip; random resized crop removed','new_validation_geometry':'whole-region letterbox 224x224; no center crop'})
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key,title in zip(axes,['accuracy','top5','macro_f1'],['Top-1 accuracy','Top-5 accuracy','Macro-F1']):
        ax.bar(NAMES,[100*summary[m][key]['mean'] for m in MODES],yerr=[100*summary[m][key]['std'] for m in MODES],color=COLORS,capsize=4)
        ax.set(title=title,ylabel='Percent',ylim=(0,100));ax.tick_params(axis='x',labelsize=9);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Same validation split: mean and sample SD across 3 training seeds');fig.tight_layout();fig.savefig(OUT/'performance_comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5));x=np.arange(4);width=.25
    for j,(mode,color,name) in enumerate(zip(MODES,COLORS,NAMES)):
        ax.bar(x+(j-1)*width,[100*r[mode+'_mean'] for r in binned],width,yerr=[100*r[mode+'_std'] for r in binned],color=color,label=name,capsize=3)
    ax.set(xticks=x,xticklabels=[r['bird_box_area_group']+f"\n(n={r['images']})" for r in binned],xlabel='Bird box area / original image area',ylabel='Validation Top-1 (%)',ylim=(0,100));ax.legend();ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'bird_size_comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for mode,color,name in zip(MODES,COLORS,NAMES):
        histories=[]
        for seed in [42,43,44]:
            directory=f'letterbox_seed{seed}' if mode=='letterbox' else 'baseline_full' if (mode,seed)==('full',42) else 'crop_margin010' if (mode,seed)==('crop',42) else f'comparison_{mode}_seed{seed}'
            histories.append([float(r['val_accuracy'])*100 for r in csv.DictReader((ROOT/'runs'/directory/'history.csv').open())])
        v=np.array(histories);x=np.arange(1,11);mean=v.mean(0);sd=v.std(0,ddof=1)
        ax.plot(x,mean,color=color,label=name);ax.fill_between(x,mean-sd,mean+sd,color=color,alpha=.12)
    ax.set(xlabel='Epoch',ylabel='Validation Top-1 (%)',title='Mean and sample SD across 3 training seeds');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'validation_curves.png',dpi=180);plt.close(fig)
    # Select diverse classes for both improvements and regressions, seed 42.
    selected=[]
    for improves in [True,False]:
        seen=set();count=0
        for row in rows:
            a=pm[(42,'crop')][row['id']]==row['label'];b=pm[(42,'letterbox')][row['id']]==row['label']
            if (not a and b if improves else a and not b) and row['label'] not in seen:
                selected.append(row);seen.add(row['label']);count+=1
                if count==3:break
    fig,axes=plt.subplots(len(selected),3,figsize=(12,3*len(selected)),squeeze=False)
    for i,row in enumerate(selected):
        for j,mode in enumerate(MODES):
            tensor=Birds(data,[row],False,mode!='full',.1,'letterbox' if mode=='letterbox' else 'legacy')[0][0]
            array=(tensor.numpy().transpose(1,2,0)*np.array([.229,.224,.225])+np.array([.485,.456,.406])).clip(0,1)
            ax=axes[i,j];ax.imshow(array);ax.axis('off');pred=pm[(42,mode)][row['id']]
            ax.set_title(NAMES[j]+': '+classes[pred][4:].replace('_',' '),fontsize=8,color='green' if pred==row['label'] else 'red')
        axes[i,0].set_ylabel('True: '+classes[row['label']][4:].replace('_',' '))
    fig.suptitle('Actual network inputs, seed 42: letterbox fixes then regressions vs center crop\nSelected examples; green = correct, red = incorrect',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(OUT/'qualitative_examples.png',dpi=160);plt.close(fig)
    save_csv('example_ids.csv',[{'image_id':r['id'],'true_class':classes[r['label']],'path':r['path']} for r in selected])
    lines=['保留完整鸟区域：等比例缩放与填充实验','', '设置：固定 split-seed=42；训练 seed=42、43、44；ResNet-18；训练4794张、验证1200张；10轮，前2轮冻结主干；batch32；AdamW主干学习率0.0001、分类头0.001；weight decay0.0001；混合精度。每次选择验证Top-1最高的checkpoint。', '新处理：同样的真实边界框、每侧10%边缘；最长边缩放为224，保持比例，居中填充至224×224。填充RGB=(124,116,104)，接近ImageNet均值。训练保留水平翻转，取消随机裁剪；验证取消中心裁剪。旧两组结果直接读取已保存记录，未覆盖。','', '结果（3个训练种子的均值±样本标准差，不是置信区间）：']
    for mode,name in zip(MODES,['整图','中心裁剪','完整区域缩放填充']):
        lines.append(name+'：'+'；'.join(f"{title} {100*summary[mode][key]['mean']:.2f}% ± {100*summary[mode][key]['std']:.2f}%" for key,title in [('accuracy','Top-1'),('top5','Top-5'),('macro_f1','Macro-F1')]))
    for ref,name in [('full','整图'),('crop','中心裁剪')]:
        lines.append(f"缩放填充相对{name}的平均Top-1差值：{100*(summary['letterbox']['accuracy']['mean']-summary[ref]['accuracy']['mean']):+.2f}个百分点。")
    lines+=['','限制：此实验同时取消训练随机裁剪、替换验证几何处理，不能把差异只归因于padding。真实位置标注属于额外监督。填充会减少鸟在224×224画布中的有效像素。官方测试集未评估；小目标分组仅33张；3个种子为初步稳定性证据；CUB与ImageNet可能重叠。案例按结果选择，不代表总体分布。','', '文件：summary.json、validation_results.csv、paired_differences.csv、bird_size_analysis.csv、validation_predictions.csv、三组性能图、验证曲线和实际网络输入案例。','复现：在项目目录运行 run_letterbox.ps1；完成的训练会跳过。']
    (OUT/'experiment_report.txt').write_text('\n'.join(lines),encoding='utf-8-sig')
    print('LETTERBOX COMPARISON COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':main()
