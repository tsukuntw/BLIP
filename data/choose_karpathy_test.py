import os
import json

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

import shutil
from pathlib import Path

ann_root = '../dataset/coco/annotation/'
filenames = {'val':'coco_karpathy_val.json','test':'coco_karpathy_test.json'}

annotation = json.load(open(os.path.join(ann_root,filenames['test']),'r'))

file_list = []

file_list_colab = []

# path = Path('../dataset/coco/image/karpathy_test_far').mkdir(parents=True, exist_ok=True)
# get the select 100 images
# for item in Path(r'../dataset/coco/image/karpathy_test_far').iterdir():
#     try:
#         file_list.append('val2014/{}'.format(Path(item).name))
#         # shutil.copy(os.path.join('/../da-h/dataset/coco/image/', item) , Path(f"../da-h/dataset/coco/image/karpathy_test_images"))
#     except Exception as exc:
#         print(exc)
# print('file_list {}'.format(len(file_list)))     
# print('file_list_colab {}'.format(len(file_list_colab)))
        

# ../dataset/coco/annotation/coco_karpathy_scale_0.79_0.85_colab.json
path_colab_file = {'0.7':'../dataset/coco/annotation/coco_karpathy_scale_0.55_0.7_colab.json',
'0.85':'../dataset/coco/annotation/coco_karpathy_scale_0.7_0.85_colab.json',
'1':'../dataset/coco/annotation/coco_karpathy_scale_0.85_1_colab.json',
'occ0.7':'../dataset/coco/annotation/coco_karpathy_scale_0.55_0.7_colabocc.json',
'occ0.85':'../dataset/coco/annotation/coco_karpathy_scale_0.7_0.85_colabocc.json',
'occ1':'../dataset/coco/annotation/coco_karpathy_scale_0.85_1_colabocc.json',
'occ1_images':'../dataset/coco/annotation/coco_karpathy_scale_0_85_1_colabocc_images.json',
'yolo0.85':'../yolo/detection_coco_results_0.85.json',
'yolo0.7':'../yolo/detection_coco_results_0.7.json',
'yolo0.55':'../yolo/detection_coco_results_0.55.json',
'yoloocc0.85':'../yolo/detection_coco_results_occ_0.85_1.json',
'yoloocc0.7':'../yolo/detection_coco_results_occ_0.7_0.85.json',
'yoloocc0.55':'../yolo/detection_coco_results_occ_0.55_0.7.json',
}
# get the select from colab 100 images
path_colab = json.load(open(os.path.join(path_colab_file['occ1_images']),'r'))
for i, item in enumerate(path_colab):
    if i == 200:
        break
    file_list_colab.append('val2014/{}'.format(item)) #
print('file_num{}'.format(len(file_list_colab)))    

# load epoch results select the same with 100 images
dic2 = {} 
output2 = []   
# ../output/caption_coco_finetune09078/result/test_epoch3.json, mnt/home/da-h/Scale/output/caption_coco_finetune09078/test_split_colab_0.79_0.85_nearx_finetune09078.json
# ../LAVIS/lavis/output/BLIP2/Caption_coco/freezeqformer/20231125003/result/test_epoch2.json
# ../LAVIS/lavis/output/BLIP2/Caption_coco/freezeqformer/test_split_colab_0.55_0.7_nearx_freezeqformer.json
# stage1
# ../LAVIS/lavis/output/BLIP2/Pretrain_stage1/trainscaleqformer/20231212050/result//test_epoch1.json
# ../LAVIS/lavis/output/BLIP2/Pretrain_stage1/trainscaleqformer/test_split_colab_0.55_0.7_nearx_freezeqformer.json
# ../output/caption_coco_finetune_skip_window_0907/result/test_epoch3.json
# ../output/caption_coco_finetune_0907_large_winsize6/result/test_epoch3.json
epoch_file = json.load(open(os.path.join('../output/caption_coco_finetune_09078_winsize6/result/test_epoch3.json'),'r'))
for img_id, ann in enumerate(epoch_file):
    dic2[int(ann['image_id'])] = ann

for i in range(len(file_list_colab)):
    image_id =  int(str(file_list_colab[i].split('_')[-1].split('.')[0]))
    output2.append(dic2[image_id])
# write to file
with open(os.path.join('../output/caption_coco_finetune_09078_winsize6/', "test_split_colabocc_0.85_1_blip_images.json"),"w") as f:
            f.write(json.dumps(output2))
            
            
# load baseline results select the same with 100 images
dic3 = {} 
output3 = []   
# ../Scale/output/caption_coco_baseline/result/test_epoch0.json, ../output/caption_coco_baseline/result/baseline_test_split_colab_0.79_0.85_nearx_finetune09078.json
# ../LAVIS/lavis/output/BLIP2/Caption_coco_opt2.7b/20231128003/result/test_epochbest.json
# ../LAVIS/lavis/output/BLIP2/Caption_coco_opt2.7b/', "baseline_test_split_colab_0.55_0.7_nearx_freezeqformer.json
# stage1
# ../LAVIS/lavis/output/BLIP2/Pretrain_stage1/trainqformerfromstage1/20231214042/result/test_epoch1.json
# '../LAVIS/lavis/output/BLIP2/Pretrain_stage1/trainqformerfromstage1/', "baseline_test_split_colab_0.55_0.7_nearx_freezeqformer.json"

# baseline_file = json.load(open(os.path.join('../output/caption_coco_baseline/result/test_epoch0.json'),'r'))
# for img_id, ann in enumerate(baseline_file):
#     dic3[int(ann['image_id'])] = ann

# for i in range(len(file_list_colab)):
#     image_id =  int(str(file_list_colab[i].split('_')[-1].split('.')[0]))
#     output3.append(dic3[image_id])
# write to file
# with open(os.path.join('../output/caption_coco_baseline/result/',"baseline_test_split_colabocc_0.85_1_blip_images.json"),"w") as f:
#             f.write(json.dumps(output3))

    

# prepare gt  
# dic = {}
# output = []
        
# annotation = json.load(open(os.path.join(ann_root,filenames['test']),'r'))
# for img_id, ann in enumerate(annotation):
#     dic[ann['image']] = ann

# for i in range(len(file_list)):
#     output.append(dic[file_list[i]])
# with open(os.path.join(ann_root, "coco_karpathy_test_split_more.json"),"a") as f:
#             f.write(json.dumps(output))

    
    
