mkdir weights; cd weights; curl -O https://download.pytorch.org/models/vgg16-397923af.pth; cd ..

mkdir coco; cd coco
curl -O http://images.cocodataset.org/zips/train2017.zip; unzip train2017.zip
curl -O http://images.cocodataset.org/zips/val2017.zip; unzip val2017.zip
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip; unzip annotations_trainval2017.zip
cd ..
