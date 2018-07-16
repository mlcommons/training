# Get VGG model
cd ./ssd; curl -O https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
python3 base_model.py; rm vgg16_reducedfc.pth; cd ..
