from pycocotools.coco import COCO
annFile = "vg_objects.json"
coco = COCO(annFile)
id2label = {
    k - 1: v["name"] for k, v in coco.cats.items()
}
print(id2label)