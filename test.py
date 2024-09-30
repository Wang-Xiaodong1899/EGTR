from glob import glob
import json

import os

import torch
from PIL import Image, ImageDraw, ImageFont

from model.deformable_detr import DeformableDetrConfig, DeformableDetrFeatureExtractor
from model.egtr import DetrForSceneGraphGeneration

# config
architecture = "./deformable-detr"
min_size = 800
max_size = 1333
artifact_path = "./version_0"

# for VG
with open(f"VG/rel.json", "r") as f:
    rel = json.load(f)
id2relation = rel["rel_categories"][1:]

from pycocotools.coco import COCO
annFile = "vg_objects.json"
coco = COCO(annFile)
id2label = {
    k - 1: v["name"] for k, v in coco.cats.items()
}

# feature extractor
feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(
    architecture, size=min_size, max_size=max_size
)

# model
config = DeformableDetrConfig.from_pretrained(artifact_path)
model = DetrForSceneGraphGeneration.from_pretrained(
    architecture, config=config, ignore_mismatched_sizes=True
)
ckpt_path = sorted(
    glob(f"{artifact_path}/checkpoints/epoch=*.ckpt"),
    key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
)[-1]
state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
for k in list(state_dict.keys()):
    state_dict[k[6:]] = state_dict.pop(k)  # "model."



model.load_state_dict(state_dict)
model.cuda()
model.eval()


def inference(image_path = "/root/egtr/v_0y_5NIIvUzI-Scene-002/c01_0001.jpeg"):
    # inference image
    pil = Image.open(image_path)
    image = feature_extractor(pil, return_tensors="pt")

    # output
    outputs = model(
        pixel_values=image['pixel_values'].cuda(), 
        pixel_mask=image['pixel_mask'].cuda(), 
        output_attention_states=True
    )

    pred_logits = outputs['logits'][0]
    obj_scores, pred_classes = torch.max(pred_logits.softmax(-1), -1)
    pred_boxes = outputs['pred_boxes'][0]

    pred_connectivity = outputs['pred_connectivity'][0]
    pred_rel = outputs['pred_rel'][0]
    pred_rel = torch.mul(pred_rel, pred_connectivity)

    # get valid objects and triplets
    obj_threshold = 0.4
    valid_obj_indices = (obj_scores >= obj_threshold).nonzero()[:, 0]

    valid_obj_classes = pred_classes[valid_obj_indices] # [num_valid_objects]
    valid_obj_boxes = pred_boxes[valid_obj_indices] # [num_valid_objects, 4]

    valid_obj_classes = valid_obj_classes.detach().cpu().numpy()
    bboxes = valid_obj_boxes.detach().cpu().numpy()

    # valid_obj_classes = [id2label[idx] for idx in valid_obj_classes]


    # print(valid_obj_classes)

    # Draw
    draw = ImageDraw.Draw(pil)
    i = 0
    colors = ['red', 'blue', 'orange', 'green', 'yellow', 'purple']
    total_objs = []
    obj_sum = {}
    #这里采集的一般是不重复的
    for bbox, obj in zip(bboxes, valid_obj_classes):
        x_center, y_center, width, height = bbox
        x_max = int((x_center + width/2) * pil.width)
        y_max = int((y_center + height/2) * pil.height)
        x_min = int((x_center - width/2) * pil.width)
        y_min = int((y_center - height/2) * pil.height)
        obj = id2label[obj]
        if obj not in obj_sum:
            obj_sum[obj] = 0
        else:
            obj_sum[obj] = obj_sum[obj] + 1
        total_objs.append(obj+f"{obj_sum[obj]}")
        text_width, text_height = 10, 10
        text_x = max(x_min + text_width + 5, 0)
        text_y = y_min + text_height + 5
        draw.text((text_x, text_y), str(obj+f"{obj_sum[obj]}"), fill=colors[i])
        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=colors[i], fill=None, width=3)
        i = (i + 1)%6
    pil.save(f"{image_path.split('.')[0]}_box.png")
    print(f"total obj: {total_objs}")
    # import pdb; pdb.set_trace()
    rel_threshold = 1e-7
    valid_triplets = (pred_rel[valid_obj_indices][:, valid_obj_indices] >= rel_threshold).nonzero() # [num_valid_triplets, 3]
    # import pdb; pdb.set_trace()
    valid_triplets = valid_triplets.detach().cpu().numpy()

    # import pdb; pdb.set_trace()

    # 提取 subject, object 和 relation 的索引
    subjects = valid_triplets[:, 0]
    objects = valid_triplets[:, 1]
    relations = valid_triplets[:, 2]

    # 获取对应的关系置信度
    relation_confidences = pred_rel[valid_obj_indices][subjects, objects, relations]

    # 合并结果
    valid_triplets_with_confidence = [(sub.item(), obj.item(), rel.item(), conf.item()) for sub, obj, rel, conf in zip(subjects, objects, relations, relation_confidences)]

    triplets, triplet_boxes = [], []
    max_conf = {}

    for sub, obj, rel, conf in valid_triplets_with_confidence:
        triplet_boxes.append([bboxes[sub], bboxes[obj]])
        # sub_label = id2label[valid_obj_classes[sub]]
        # obj_label = id2label[valid_obj_classes[obj]]
        sub_label = total_objs[sub]
        obj_label = total_objs[obj]
        rel_label = id2relation[rel]

        # 如果当前置信度大于阈值才考虑
        if conf > rel_threshold:
            key = (obj_label, sub_label)
            
            # 如果key还未出现或者当前置信度比之前的大，更新
            if key not in max_conf or conf > max_conf[key][1]:
                max_conf[key] = (rel_label, conf)

    # 遍历字典，生成最终的triplets
    for (obj_label, sub_label), (rel_label, conf) in max_conf.items():
        if obj_label != sub_label:
            triplets.append([obj_label, rel_label, sub_label])
            print(triplets[-1])

    import pdb; pdb.set_trace()


    import networkx as nx
    import matplotlib.pyplot as plt
    
    plt.clf()

    G = nx.DiGraph()

    for triple in triplets:
        G.add_edge(triple[0], triple[2], label=triple[1])

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, arrows=True)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Directed Graph from Triples")
    plt.savefig(f"{image_path.split('.')[0]}_sg.png")


# root = "/root/egtr/v_0y_5NIIvUzI-Scene-002"
# files = os.listdir(root)

# for file in files:
#     if file.split('.')[-1] == "jpeg":
#         inference(os.path.join(root, file))

inference("tantic.jpg")