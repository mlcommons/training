mkdir detectron/lib/datasets/data/coco
mv coco_annotations_minival.tgz detectron/lib/datasets/data/coco
mv train2014.zip detectron/lib/datasets/data/coco
mv val2014.zip detectron/lib/datasets/data/coco
mv annotations_trainval2014.zip detectron/lib/datasets/data/coco

cd detectron/lib/datasets/data/coco
dtrx --one=here coco_annotations_minival.tgz
dtrx --one=here annotations_trainval2014.zip
mv annotations.1/* annotations/

dtrx train2014.zip
mv train2014/ coco_train2014/
dtrx val2014.zip
mv val2014/ coco_val2014/
