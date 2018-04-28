cd detectron
time stdbuf -o 0 \
  python tools/train_net.py --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml \
  --box_min_ap 0.377 --mask_min_ap 0.339 \
  --seed 3 | tee run.log
