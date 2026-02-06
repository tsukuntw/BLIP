'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.blips import blip_decoder
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.utils import save_result


    
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from torchvision.datasets.utils import download_url

def coco_caption_eval(coco_gt_root, results_file, split):
    urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json',
            'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json'}
    filenames = {'val':'coco_karpathy_val_gt.json','test':'coco_karpathy_test_gt.json'}    
    
    download_url(urls[split],coco_gt_root)
    annotation_file = os.path.join(coco_gt_root,filenames[split])
    
    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()
    
    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
        
    object_f1 = [
        eval_res['SPICE']['Object']['f']
        for eval_res in coco_eval.imgToEval.values()
    ]
    print(f'SPICE F1 (Object): {np.mean(object_f1):.3f}')
     
    
    return coco_eval

#
load_file_baseline = {
'blip':'./Scale/output/caption_coco_baseline/result/test_epoch0.json',
'bliplarge':'./Scale/output/caption_coco_baseline_large/result/test_epoch0.json',
'blipcolabocc85':'./Scale/output/caption_coco_baseline/result/baseline_test_split_colabocc_0.85_1_blip_images.json',
'blipcolabocc70':'./Scale/output/caption_coco_baseline/result/baseline_test_split_colabocc_0.7_0.85_blip.json',
'blipcolabocc55':'./Scale/output/caption_coco_baseline/result/baseline_test_split_colabocc_0.55_0.7_blip.json',
'bliplargecolabocc85':'./Scale/output/caption_coco_baseline_large/result/baseline_large_test_split_colabocc_0.85_1_blip.json',
'bliplargecolabocc70':'./Scale/output/caption_coco_baseline_large/result/baseline_large_test_split_colabocc_0.7_0.85_blip.json',
'bliplargecolabocc55':'./Scale/output/caption_coco_baseline_large/result/baseline_large_test_split_colabocc_0.55_0.7_blip.json',
'blipcolabocc85maxpooling':'./Scale/output/caption_coco_baseline/result/baseline_test_split_colabocc_0.85_1_blip_images.json',
'blipmaxpooling':'./Scale/output/caption_coco_baseline/result/test_epoch0.json',
'blipcolabocc85avgpooling':'./Scale/output/caption_coco_baseline/result/baseline_test_split_colabocc_0.85_1_blip_images.json',
'blipavgpooling':'./Scale/output/caption_coco_baseline/result/test_epoch0.json',
'blipcolabocc85local':'./Scale/output/caption_coco_baseline/result/baseline_test_split_colabocc_0.85_1_blip_images.json',
'bliplocal':'./Scale/output/caption_coco_baseline/result/test_epoch0.json',
'blipcolabocc85swin':'./Scale/output/caption_coco_baseline/result/baseline_test_split_colabocc_0.85_1_blip.json',
'blipswin':'./Scale/output/caption_coco_baseline/result/test_epoch0.json',
'blipcolabocc85merging':'./Scale/output/caption_coco_baseline/result/baseline_test_split_colabocc_0.85_1_blip_images.json',
'blipmerging':'./Scale/output/caption_coco_baseline/result/test_epoch0.json',
'blipcolabocc85windowattention':'./Scale/output/caption_coco_baseline/result/baseline_test_split_colabocc_0.85_1_blip_images.json',

}

load_file_our = {
'blip':'./Scale/output/caption_coco_finetune_skip_window_0907_winsize6/result/test_epoch3.json',
'bliplarge':'./Scale/output/caption_coco_finetune_0907_large_winsize6/result/test_epoch3.json',
'blipcolabocc85':'./Scale/output/caption_coco_finetune_skip_window_0907_winsize6/test_split_colabocc_0.85_1_blip_images.json',
'blipcolabocc70':'./Scale/output/caption_coco_finetune_skip_window_0907_winsize6/test_split_colabocc_0.7_0.85_blip.json',
'blipcolabocc55':'./Scale/output/caption_coco_finetune_skip_window_0907_winsize6/test_split_colabocc_0.55_0.7_blip.json',
'bliplargecolabocc85':'./Scale/output/caption_coco_finetune_0907_large_winsize6/test_split_colabocc_0.85_1_blip.json',
'bliplargecolabocc70':'./Scale/output/caption_coco_finetune_0907_large_winsize6/test_split_colabocc_0.7_0.85_blip.json',
'bliplargecolabocc55':'./Scale/output/caption_coco_finetune_0907_large_winsize6/test_split_colabocc_0.55_0.7_blip.json',
'blipcolabocc85maxpooling':'./Scale/output/caption_coco_finetune_skip_window_simplemax_0907_winsize6/test_split_colabocc_0.85_1_blip_images.json',
'blipmaxpooling':'./Scale/output/caption_coco_finetune_skip_window_simplemax_0907_winsize6/result/test_epoch3.json',
'blipcolabocc85avgpooling':'./Scale/output/caption_coco_finetune_skip_window_simpleavg_0907_winsize6/test_split_colabocc_0.85_1_blip_images.json',
'blipavgpooling':'./Scale/output/caption_coco_finetune_skip_window_simpleavg_0907_winsize6/result/test_epoch2.json',
'blipcolabocc85local':'./Scale/output/caption_coco_finetune_local_0907_winsize6/test_split_colabocc_0.85_1_blip_images.json',
'bliplocal':'./Scale/output/caption_coco_finetune_local_0907_winsize6/result/test_epoch4.json',
'blipcolabocc85swin':'./Scale/output/caption_coco_finetune_09078_winsize6/test_split_colabocc_0.85_1_blip.json',
'blipswin':'./Scale/output/caption_coco_finetune_09078_winsize6/result/test_epoch3.json',
'blipcolabocc85merging':'./Scale/output/caption_coco_finetune_merging_0907/test_split_colabocc_0.85_1_blip_images.json',
'blipmerging':'./Scale/output/caption_coco_finetune_merging_0907/result/test_epoch3.json',
'blipcolabocc85windowattention':'./Scale/output/caption_coco_finetune_09078_winsize6/test_split_colabocc_0.85_1_blip_images.json',
}

def mean_metrics(image_dict_all):
    metrics = {
        "baseline": {
            "C": [],
            "S": [],
            "SPICE Object f1": [],
            "ObjectExcessRate": []
        },
        "test": {
            "C": [],
            "S": [],
            "SPICE Object f1": [],
            "ObjectExcessRate": []
        }
    }

    for item in image_dict_all:
        for split in ["baseline", "test"]:
            metrics[split]["C"].append(item[split]["C"])
            metrics[split]["S"].append(item[split]["S"])
            metrics[split]["SPICE Object f1"].append(
                item[split]["SPICE Object f1"]
            )
            metrics[split]["ObjectExcessRate"].append(
                item[split]["ObjectExcessRate"]
            )

    mean_results = {}
    for split in metrics:
        mean_results[split] = {
            k: np.mean(v) if len(v) > 0 else 0.0
            for k, v in metrics[split].items()
        }

    return mean_results


def main(args, config):
        test_baseline_file = load_file_baseline['blipcolabocc85windowattention']  # baseline results json
        test_result_file = load_file_our['blipcolabocc85windowattention']  # final results json
        
        baseline_test = coco_caption_eval(config['coco_gt_root'],test_baseline_file, 'test') 
        coco_test = coco_caption_eval(config['coco_gt_root'],test_result_file, 'test') 
        
                      
        log_stats = {**{f'baseline_test_{k}': v for k, v in baseline_test.eval.items()}, 
                     **{f'test_{k}': v for k, v in coco_test.eval.items()},                       
                    }
        with open(os.path.join(args.output_dir, "evaluate_test_blip.txt"),"a") as f:
            f.write(json.dumps({'start': 'baseline_file {}, test_ourfile_{}'.format(test_baseline_file, test_result_file)}) + "\n") 
            f.write(json.dumps(log_stats, indent=2) + "\n")  
            f.write(json.dumps('Successful') + "\n")     
            
        # https://github.com/tylin/coco-caption/blob/master/cocoEvalCapDemo.ipynb
        # demo how to use evalImgs to retrieve low score result
        # dict {id: spice}
        baseline_evals = {eva['image_id']: [eva['Bleu_4'], eva['CIDEr'], eva['SPICE']['All']['f'], eva['SPICE']['Object']['f'],
        eva['SPICE']['ObjectAnalysis']] for eva in baseline_test.evalImgs}
        test_evals = {eva['image_id']: [eva['Bleu_4'], eva['CIDEr'], eva['SPICE']['All']['f'], eva['SPICE']['Object']['f'],
        eva['SPICE']['ObjectAnalysis']] for eva in coco_test.evalImgs}
        # print 'res captions'
        
        # image_id_dict = {image_id: {'baseline SPICE': '{:.3f}'.format(baseline_evals[image_id][2]), 
        #                   'test SPICE': '{:.3f}'.format(test_evals[image_id][2])} 
        #                   for image_id in list(test_evals.keys())
        #                   # if all([x > y for x, y in zip(test_evals[image_id][2], baseline_evals[image_id][2])]) # all metrics better
        #                   
        # }
                                                     
        
        with open(os.path.join(args.output_dir, "baseline_evals_blip.txt"),"a") as f1:
            f1.write(json.dumps(baseline_test.evalImgs) + "\n")  
            
        with open(os.path.join(args.output_dir, "test_evals_blip.txt"),"a") as f2:
            f2.write(json.dumps(coco_test.evalImgs) + "\n")   

        
        image_baseline_dict = json.load(open(os.path.join(test_baseline_file),'r'))
        image_test_dict = json.load(open(os.path.join(test_result_file),'r'))
        
        image_cap_baseline = {item['image_id']: item['caption'] for item in image_baseline_dict
                   if item['image_id'] in list(baseline_evals.keys())}
        image_cap_test = {item['image_id']: item['caption'] for item in image_test_dict
                   if item['image_id'] in list(test_evals.keys())}
                   
        image_dict_all = [{'image_id': image_id, 'baseline': {'C': baseline_evals[image_id][1],
                                                              'S': baseline_evals[image_id][2],
                                                              'SPICE Object f1': baseline_evals[image_id][3],
                                                              'ObjectExcessRate': baseline_evals[image_id][4]['hallucination_rate']}, 
                                                              'test': {'C': test_evals[image_id][1], 
                                                              'S': test_evals[image_id][2],
                                                              'SPICE Object f1': test_evals[image_id][3],
                                                              'ObjectExcessRate': test_evals[image_id][4]['hallucination_rate']}} 
                                                              for image_id in list(test_evals.keys())]
        # 
        mean_res = mean_metrics(image_dict_all)
        print("Baseline mean:")
        for k, v in mean_res["baseline"].items():
            print(f"  {k}: {v:.4f}")
        
        print("Test mean:")
        for k, v in mean_res["test"].items():
            print(f"  {k}: {v:.4f}")
                   
        image_cap_dict = [{'image_id': image_id, 'baseline': {'caption': image_cap_baseline[image_id], 
                                                              'C': '{:.3f}'.format(baseline_evals[image_id][1]),
                                                              'S': '{:.3f}'.format(baseline_evals[image_id][2]),
                                                              'SPICE Object f1': '{:.3f}'.format(baseline_evals[image_id][3]),
                                                              'ObjectAnalysis': '{}'.format(baseline_evals[image_id][4])}, 
                                                 'test': {'caption': image_cap_test[image_id], 
                                                          'C': '{:.3f}'.format(test_evals[image_id][1]), 
                                                          'S': '{:.3f}'.format(test_evals[image_id][2]),
                                                          'SPICE Object f1': '{:.3f}'.format(test_evals[image_id][3]),
                                                          'ObjectAnalysis': '{}'.format(test_evals[image_id][4]),}} for image_id in list(test_evals.keys()) if len(test_evals[image_id][4]['hallucinated']) < len(baseline_evals[image_id][4]['hallucinated'])]
        
        
        with open(os.path.join(args.output_dir, "coco_test_blip.txt"),"a") as f:
            f.write(json.dumps(image_cap_dict, indent=2) + "\n") 
            print(f'length image_cap_dict {len(image_cap_dict)}')
            
        
        # imgId = best_evals[0]['image_id']
        
        
        # annIds = coco.getAnnIds(imgIds=imgId)
        # anns = coco.loadAnns(annIds)
        # coco.showAnns(anns)
        
        # print '\n'
        # print 'generated caption (CIDEr score %0.1f)'%(evals[0]['CIDEr'])
        
        # annIds = cocoRes.getAnnIds(imgIds=imgId)
        # anns = cocoRes.loadAnns(annIds)
        # coco.showAnns(anns)
        
        # img = coco.loadImgs(imgId)[0]
        # I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
        # plt.imshow(I)
        # plt.axis('off')
        # plt.show()             


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/caption_coco_load_pretrain_scale.yaml')
    parser.add_argument('--output_dir', default='output/caption_coco_analyse/overall') # output/caption_coco_analyse       
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    # yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
