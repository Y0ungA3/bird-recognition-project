"""Paired validation comparison. Never reads official test images."""
import csv, json, math, os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache'/'matplotlib'))
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader
from torchvision import models
from train import Birds, make_splits, read_pairs, run_epoch

OUT=ROOT/'runs'/'comparison'

def write_csv(name,rows):
    if not rows: return
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=ROOT/'data'/'CUB_200_2011'
    val=make_splits(data,42)['val']
    classes=list(read_pairs(data/'classes.txt').values())
    meta=[]
    for row in val:
        with Image.open(data/'images'/row['path']) as im: width,height=im.size
        x,y,w,h=row['bbox']
        meta.append({**row,'bird_area_ratio':w*h/(width*height)})
    all_results=[]; all_preds=[]; pred_map={}; curves={};reference=None
    for seed in [42,43,44]:
        for mode in ['full','crop']:
            name='baseline_full' if (seed,mode)==(42,'full') else 'crop_margin010' if (seed,mode)==(42,'crop') else f'comparison_{mode}_seed{seed}'
            path=ROOT/'runs'/name
            assert (path/'completed.json').exists(),f'Incomplete run: {name}'
            ckpt=torch.load(path/'best.pt',map_location='cpu',weights_only=False)
            cfg=ckpt['config']
            controlled={k:cfg[k] for k in ['epochs','freeze_epochs','batch_size','workers','margin']}
            controlled['split_sha256']=ckpt['split_sha256']
            if reference is None: reference=controlled
            assert controlled==reference, f'Unmatched experimental settings: {name}'
            assert cfg['seed']==seed and cfg['crop']==(mode=='crop') and not cfg['smoke']
            assert json.loads((path/'splits.json').read_text())['val']==[r['id'] for r in val]
            model=models.resnet18(weights=None);model.fc=torch.nn.Linear(512,200)
            model.load_state_dict(ckpt['model']);model.cuda().eval()
            loader=DataLoader(Birds(data,val,False,mode=='crop',.1),batch_size=32,shuffle=False,num_workers=0,pin_memory=True)
            with torch.inference_mode(): result,cm,preds=run_epoch(model,loader,torch.device('cuda'),torch.nn.CrossEntropyLoss())
            result={'seed':seed,'mode':mode,'best_epoch':ckpt['epoch'],**result}
            (path/'validation_metrics.json').write_text(json.dumps(result,indent=2))
            all_results.append(result);pred_map[(seed,mode)]={i:p for i,y,p in preds}
            for i,y,p in preds: all_preds.append({'seed':seed,'mode':mode,'image_id':i,'true_class_0based':y,'predicted_class_0based':p,'correct':int(y==p)})
            curves[(seed,mode)]=list(csv.DictReader((path/'history.csv').open()))
            print('Validated:',result,flush=True)
            del model,loader;torch.cuda.empty_cache()
    write_csv('validation_results.csv',all_results)
    write_csv('validation_predictions.csv',all_preds)
    summary={}; paired=[]
    for mode in ['full','crop']:
        summary[mode]={metric:{'mean':float(np.mean([r[metric] for r in all_results if r['mode']==mode])),'std':float(np.std([r[metric] for r in all_results if r['mode']==mode],ddof=1))} for metric in ['accuracy','top5','macro_f1']}
    for seed in [42,43,44]:
        full=next(r for r in all_results if r['seed']==seed and r['mode']=='full')
        crop=next(r for r in all_results if r['seed']==seed and r['mode']=='crop')
        gains=losses=0
        for r in meta:
            a=pred_map[(seed,'full')][r['id']]==r['label'];b=pred_map[(seed,'crop')][r['id']]==r['label']
            gains+=int(b and not a);losses+=int(a and not b)
        paired.append({'seed':seed,'full_top1':full['accuracy'],'crop_top1':crop['accuracy'],'delta_percentage_points':100*(crop['accuracy']-full['accuracy']),'crop_fixes_full_errors':gains,'crop_breaks_full_correct':losses})
    write_csv('paired_differences.csv',paired)
    bins=[(0,.1,'<10%'),(.1,.3,'10-30%'),(.3,.6,'30-60%'),(.6,float('inf'),'>=60%')]
    binned=[]
    for low,high,label in bins:
        rows=[r for r in meta if low<=r['bird_area_ratio']<high]
        if not rows: continue
        row={'bird_area_group':label,'images':len(rows)}
        for mode in ['full','crop']:
            values=[np.mean([pred_map[(s,mode)][r['id']]==r['label'] for r in rows]) for s in [42,43,44]]
            row[mode+'_accuracy_mean']=float(np.mean(values));row[mode+'_accuracy_std']=float(np.std(values,ddof=1))
        row['delta_percentage_points']=100*(row['crop_accuracy_mean']-row['full_accuracy_mean']);binned.append(row)
    write_csv('bird_size_analysis.csv',binned)
    class_results=[]
    for c,name in enumerate(classes):
        rows=[r for r in meta if r['label']==c]
        a=np.mean([pred_map[(s,'full')][r['id']]==c for s in [42,43,44] for r in rows])
        b=np.mean([pred_map[(s,'crop')][r['id']]==c for s in [42,43,44] for r in rows])
        class_results.append({'class':name,'validation_images':len(rows),'full_accuracy_mean':float(a),'crop_accuracy_mean':float(b),'delta_percentage_points':float(100*(b-a))})
    write_csv('per_class_results.csv',sorted(class_results,key=lambda r:r['delta_percentage_points'],reverse=True))
    summary.update({'paired':paired,'mean_paired_delta_percentage_points':float(np.mean([r['delta_percentage_points'] for r in paired])),'split_sha256':reference['split_sha256'],'training_seeds':[42,43,44],'split_seed':42,'train_images':4794,'validation_images':1200,'official_test_evaluated':False,'crop_margin_per_side':.1,'bird_size_analysis':binned})
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    # Error bars are sample SD across training seeds, not confidence intervals.
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,metric,title in zip(axes,['accuracy','top5','macro_f1'],['Top-1 accuracy','Top-5 accuracy','Macro-F1']):
        ax.bar(['Full image','Bird crop'],[100*summary[m][metric]['mean'] for m in ['full','crop']],yerr=[100*summary[m][metric]['std'] for m in ['full','crop']],capsize=5,color=['#4477aa','#ee7733'])
        ax.set(title=title,ylabel='Percent',ylim=(0,100));ax.grid(axis='y',alpha=.2)
    fig.suptitle('Validation results: mean and sample SD across 3 training seeds')
    fig.tight_layout();fig.savefig(OUT/'performance_comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for mode,color in [('full','#4477aa'),('crop','#ee7733')]:
        values=np.array([[float(r['val_accuracy']) for r in curves[(s,mode)]] for s in [42,43,44]])*100
        mean=values.mean(0);std=values.std(0,ddof=1);x=np.arange(1,11)
        ax.plot(x,mean,label=mode,color=color);ax.fill_between(x,mean-std,mean+std,color=color,alpha=.15)
    ax.set(xlabel='Epoch',ylabel='Validation Top-1 (%)',title='Same validation split; shading = sample SD across seeds');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'validation_curves.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));x=np.arange(len(binned));width=.35
    for offset,mode,color in [(-width/2,'full','#4477aa'),(width/2,'crop','#ee7733')]:
        ax.bar(x+offset,[100*r[mode+'_accuracy_mean'] for r in binned],width,yerr=[100*r[mode+'_accuracy_std'] for r in binned],label=mode,color=color,capsize=3)
    ax.set(xticks=x,xticklabels=[r['bird_area_group']+f"\n(n={r['images']})" for r in binned],xlabel='Annotated bird box area / original image area',ylabel='Validation Top-1 (%)',ylim=(0,100),title='Performance by bird size: mean and sample SD across 3 seeds');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'bird_size_comparison.png',dpi=180);plt.close(fig)
    fixed=[r for r in meta if pred_map[(42,'full')][r['id']]!=r['label'] and pred_map[(42,'crop')][r['id']]==r['label']][:3]
    broken=[r for r in meta if pred_map[(42,'full')][r['id']]==r['label'] and pred_map[(42,'crop')][r['id']]!=r['label']][:3]
    selected=fixed+broken
    if selected:
        fig,axes=plt.subplots(len(selected),3,figsize=(12,3*len(selected)),squeeze=False)
        for idx,row in enumerate(selected):
            with Image.open(data/'images'/row['path']) as im: axes[idx,0].imshow(im.convert('RGB'))
            x,y,w,h=row['bbox'];axes[idx,0].add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='red',linewidth=1.5))
            axes[idx,0].set_title('True: '+classes[row['label']][4:].replace('_',' '),fontsize=9)
            for col,mode in [(1,'full'),(2,'crop')]:
                tensor=Birds(data,[row],False,mode=='crop',.1)[0][0]
                array=(tensor.numpy().transpose(1,2,0)*np.array([.229,.224,.225])+np.array([.485,.456,.406])).clip(0,1)
                axes[idx,col].imshow(array)
                pred=pred_map[(42,mode)][row['id']]
                axes[idx,col].set_title(mode+': '+classes[pred][4:].replace('_',' '),fontsize=9,color='green' if pred==row['label'] else 'red')
            for ax in axes[idx]:ax.axis('off')
        fig.suptitle('Selected outcome examples, seed 42: crop improvements then regressions\nLeft: original and box. Middle/right: actual 224x224 evaluation inputs.',fontsize=11)
        fig.tight_layout(rect=(0,0,1,.97));fig.savefig(OUT/'qualitative_examples.png',dpi=160);plt.close(fig)
        write_csv('qualitative_example_ids.csv',[{'image_id':r['id'],'path':r['path'],'true_class':classes[r['label']]} for r in selected])
    lines=['整图与鸟区域裁剪对照实验','', '实验条件：ResNet-18 ImageNet 预训练；200 类；训练 4794 张、验证 1200 张。固定数据划分 seed=42；训练随机种子 42、43、44。每组训练 10 轮，前 2 轮冻结主干；batch size 32；其余设置完全相同。每次运行使用验证集 Top-1 最佳 checkpoint。','裁剪条件：使用真实标注边界框，每侧扩展原框宽/高的 10%，再使用同样的训练和验证变换。','', '结果（均值 ± 样本标准差；标准差不是置信区间）：']
    for mode,label in [('full','整图'),('crop','裁剪')]:
        lines.append(label+'：'+ '；'.join(f"{title} {100*summary[mode][m]['mean']:.2f}% ± {100*summary[mode][m]['std']:.2f}%" for m,title in [('accuracy','Top-1'),('top5','Top-5'),('macro_f1','Macro-F1')]))
    lines+=['',f"裁剪相对整图的平均 Top-1 差值：{summary['mean_paired_delta_percentage_points']:+.2f} 个百分点。"]
    for r in paired:lines.append(f"seed {r['seed']}：整图 {100*r['full_top1']:.2f}%，裁剪 {100*r['crop_top1']:.2f}%，差值 {r['delta_percentage_points']:+.2f} 个百分点；纠正 {r['crop_fixes_full_errors']} 张，反而错判 {r['crop_breaks_full_correct']} 张。")
    lines+=['','按鸟占图比例分析（3 个种子的平均 Top-1）：']
    for r in binned:lines.append(f"{r['bird_area_group']}，n={r['images']}：整图 {100*r['full_accuracy_mean']:.2f}%，裁剪 {100*r['crop_accuracy_mean']:.2f}%，差值 {r['delta_percentage_points']:+.2f} 个百分点。")
    lines+=['','解释与限制：','1. 这是相同验证集上的三次训练比较，尚未评估官方测试集，不能作为最终测试结果。','2. 裁剪使用额外真实位置标注，同时改变鸟的尺度和背景内容；结果不能单独证明背景有害或模型仅依赖背景。','3. 三个随机种子只能提供初步稳定性证据；训练实现未强制所有 GPU 算子确定性。','4. 每类验证图片仅 6 张，类别级变化应谨慎解释。','5. ImageNet 预训练和 CUB 图片可能重叠，此限制对两种方法共同存在。','6. qualitative_examples.png 按结果挑选改善与退步案例，仅作解释，不代表总体分布。','7. 预处理使用 Resize(256)+CenterCrop(224)。对狭长的鸟区域，中心裁剪可能截去头、尾等局部；实际网络输入已在案例图中展示。可以在后续独立实验中比较保留完整鸟框的等比例缩放与填充，不能把本次差异直接归因于背景。','','文件：summary.json、validation_results.csv、paired_differences.csv、bird_size_analysis.csv、per_class_results.csv、validation_predictions.csv，以及四张 PNG 图。','复现：在项目目录运行 powershell -ExecutionPolicy Bypass -File run_comparison.ps1；已完成的训练会跳过，仅重新进行验证分析。']
    (OUT/'comparison_report.txt').write_text('\n'.join(lines),encoding='utf-8-sig')
    print('COMPARISON COMPLETE:',json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
