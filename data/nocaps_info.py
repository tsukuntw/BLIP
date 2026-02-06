import json
from collections import defaultdict

with open("../output/caption_coco_finetune_0907_large_winsize6_nocap/result/caption_coco_finetune_0907_large_winsize6_nocap_val.json", "r") as f:
    image_list = json.load(f)

with open("../dataset/nocaps/annotation/nocaps_val_4500_captions.json", "r") as f:
    nocaps_data = json.load(f)

annotations = defaultdict(list)
domains = defaultdict()
if "annotations" in nocaps_data:
    for ann in nocaps_data["annotations"]:
        image_id = ann["image_id"]
        caption = ann["caption"]
        annotations[image_id].append(caption)
        
for item in nocaps_data["images"]:
    image_id = item['id']
    domains[image_id]=item['domain']
        
        
image_dict = defaultdict(list)        
for item in image_list:
    image_dict[item['image_id']] = item['caption']

# total output    
output_data = defaultdict(list)
domain_list = {'in-domain': [], 'near-domain': [], 'out-domain': []}
domain_dict = {'in-domain': [], 'near-domain': [], 'out-domain': []}
i = 0
for im in nocaps_data["images"]:
    image_id = im["id"]
    image_path = 'val/' + im["file_name"]  # val/0013ea2087020901.jpg
    captions = annotations.get(image_id, [])
    domain = domains.get(image_id, []) # in out near
    print(f'{domain}')
    if captions:
        main_caption = image_dict.get(image_id, [])
        output_data[image_id].append({
            "caption": main_caption,
            "references": captions,
            "image_path": image_path
        })
        domain_dict[domain].append({
            "caption": main_caption,
            "references": captions,
            "image_path": image_path
        })
        domain_list[domain].append({"image_id": image_id, "caption": main_caption})
        i = i+1
    else:
        print(f"not found captions: {image_path}")
        
def save_domain(domain_list, is_dict = ''):
    for domain, data in domain_list.items():
        with open(f"../output/caption_coco_finetune_0907_large_winsize6_nocap/result/{domain}_{is_dict}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

# save into in-domain.json, ..., out-domain.json            
# save_domain(domain_list)
save_domain(domain_dict, 'dict')
# with open("../output/caption_coco_finetune_0907_large_winsize6_nocap/result/baseline_blip_large_nocaps_evaluate-dataset.json", "w") as f:
#     json.dump(output_data, f, indent=4)

print(f"finish, all {i}")
