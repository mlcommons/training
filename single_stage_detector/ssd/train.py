import os
from argparse import ArgumentParser
from utils import DefaultBoxes, Encoder, COCODetection
from base_model import Loss
from utils import SSDTransformer
from ssd300 import SSD300
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
import numpy as np
import mlperf_log

def parse_args():
    parser = ArgumentParser(description="Train Single Shot MultiBox Detector"
                                        " on COCO")
    parser.add_argument('--data', '-d', type=str, default='/coco',
                        help='path to test and training data files')
    parser.add_argument('--epochs', '-e', type=int, default=800,
                        help='number of epochs for training')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                        help='number of examples for each iteration')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use available GPUs')
    parser.add_argument('--seed', '-s', type=int,
                        help='manually set random seed for torch')
    parser.add_argument('--threshold', '-t', type=float, default=0.212,
                        help='stop training early at threshold')
    parser.add_argument('--iteration', type=int, default=0,
                        help='iteration to start from')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to model checkpoint file')
    parser.add_argument('--no-save', action='store_true',
                        help='save model checkpoints')
    parser.add_argument('--evaluation', nargs='*', type=int,
                        default=[120000, 160000, 180000, 200000, 220000,
                                 240000],
                        help='iterations at which to evaluate')
    return parser.parse_args()


def show_memusage(device=0):
    import gpustat
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    print("{}/{}".format(item["memory.used"], item["memory.total"]))


def dboxes300_coco():
    figsize = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    mlperf_log.ssd_print(key=mlperf_log.FEATURE_SIZES, value=feat_size)

    steps = [8, 16, 32, 64, 100, 300]
    mlperf_log.ssd_print(key=mlperf_log.STEPS, value=feat_size)

    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [21, 45, 99, 153, 207, 261, 315]
    mlperf_log.ssd_print(key=mlperf_log.SCALES, value=scales)

    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    mlperf_log.ssd_print(key=mlperf_log.ASPECT_RATIOS, value=aspect_ratios)

    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    mlperf_log.ssd_print(key=mlperf_log.NUM_DEFAULTS,
                         value=len(dboxes.default_boxes))
    return dboxes


def coco_eval(model, coco, cocoGt, encoder, inv_map, threshold,
              iteration, use_cuda=True):
    from pycocotools.cocoeval import COCOeval
    print("")
    model.eval()
    if use_cuda:
        model.cuda()
    ret = []

    overlap_threshold = 0.50
    nms_max_detections = 200
    mlperf_log.ssd_print(key=mlperf_log.NMS_THRESHOLD,
                         value=overlap_threshold)
    mlperf_log.ssd_print(key=mlperf_log.NMS_MAX_DETECTIONS,
                         value=nms_max_detections)

    start = time.time()
    for idx, image_id in enumerate(coco.img_keys):
        img, (htot, wtot), _, _ = coco[idx]

        with torch.no_grad():
            print("Parsing image: {}/{}".format(idx+1, len(coco)), end="\r")
            inp = img.unsqueeze(0)
            if use_cuda:
                inp = inp.cuda()
            ploc, plabel = model(inp)

            try:
                result = encoder.decode_batch(ploc, plabel,
                                              overlap_threshold,
                                              nms_max_detections)[0]

            except:
                #raise
                print("")
                print("No object detected in idx: {}".format(idx))
                continue

            loc, label, prob = [r.cpu().numpy() for r in result]
            for loc_, label_, prob_ in zip(loc, label, prob):
                ret.append([image_id, loc_[0]*wtot, \
                                      loc_[1]*htot,
                                      (loc_[2] - loc_[0])*wtot,
                                      (loc_[3] - loc_[1])*htot,
                                      prob_,
                                      inv_map[label_]])
    print("")
    print("Predicting Ended, total time: {:.2f} s".format(time.time()-start))

    cocoDt = cocoGt.loadRes(np.array(ret))

    E = COCOeval(cocoGt, cocoDt, iouType='bbox')
    E.evaluate()
    E.accumulate()
    E.summarize()
    print("Current AP: {:.5f} AP goal: {:.5f}".format(E.stats[0], threshold))

    # put your model back into training mode
    model.train()

    current_accuracy = E.stats[0]
    mlperf_log.ssd_print(key=mlperf_log.EVAL_SIZE, value=idx+1)
    mlperf_log.ssd_print(key=mlperf_log.EVAL_ACCURACY,
                         value={"iteration": iteration,
                                "value": current_accuracy})
    mlperf_log.ssd_print(key=mlperf_log.EVAL_TARGET, value=threshold)

    return current_accuracy>= threshold #Average Precision  (AP) @[ IoU=050:0.95 | area=   all | maxDets=100 ]



def train300_mlperf_coco(args):
    from coco import COCO
    # Check that GPUs are actually available
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)

    input_size = 300
    train_trans = SSDTransformer(dboxes, (input_size, input_size), val=False)
    val_trans = SSDTransformer(dboxes, (input_size, input_size), val=True)
    mlperf_log.ssd_print(key=mlperf_log.INPUT_SIZE, value=input_size)

    val_annotate = os.path.join(args.data, "annotations/instances_val2017.json")
    val_coco_root = os.path.join(args.data, "val2017")
    train_annotate = os.path.join(args.data, "annotations/instances_train2017.json")
    train_coco_root = os.path.join(args.data, "train2017")

    cocoGt = COCO(annotation_file=val_annotate)
    val_coco = COCODetection(val_coco_root, val_annotate, val_trans)
    train_coco = COCODetection(train_coco_root, train_annotate, train_trans)

    #print("Number of labels: {}".format(train_coco.labelnum))
    train_dataloader = DataLoader(train_coco, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # set shuffle=True in DataLoader
    mlperf_log.ssd_print(key=mlperf_log.INPUT_ORDER)
    mlperf_log.ssd_print(key=mlperf_log.INPUT_BATCH_SIZE, value=args.batch_size)


    ssd300 = SSD300(train_coco.labelnum)
    if args.checkpoint is not None:
        print("loading model checkpoint", args.checkpoint)
        od = torch.load(args.checkpoint)
        ssd300.load_state_dict(od["model"])
    ssd300.train()
    if use_cuda:
        ssd300.cuda()
    loss_func = Loss(dboxes)
    if use_cuda:
        loss_func.cuda()

    current_lr = 1e-3
    current_momentum = 0.9
    current_weight_decay = 5e-4
    optim = torch.optim.SGD(ssd300.parameters(), lr=current_lr,
                            momentum=current_momentum,
                            weight_decay=current_weight_decay)
    mlperf_log.ssd_print(key=mlperf_log.OPT_NAME, value="SGD")
    mlperf_log.ssd_print(key=mlperf_log.OPT_LR, value=current_lr)
    mlperf_log.ssd_print(key=mlperf_log.OPT_MOMENTUM, value=current_momentum)
    mlperf_log.ssd_print(key=mlperf_log.OPT_WEIGHT_DECAY,
                         value=current_weight_decay)

    print("epoch", "nbatch", "loss")

    iter_num = args.iteration
    avg_loss = 0.0
    inv_map = {v:k for k,v in val_coco.label_map.items()}

    mlperf_log.ssd_print(key=mlperf_log.TRAIN_LOOP)
    for epoch in range(args.epochs):
        mlperf_log.ssd_print(key=mlperf_log.TRAIN_EPOCH, value=epoch)
        for nbatch, (img, img_size, bbox, label) in enumerate(train_dataloader):

            if iter_num == 160000:
                current_lr = 1e-4
                print("")
                print("lr decay step #1")
                for param_group in optim.param_groups:
                    param_group['lr'] = current_lr
                mlperf_log.ssd_print(key=mlperf_log.OPT_LR,
                                     value=current_lr)

            if iter_num == 200000:
                current_lr = 1e-5
                print("")
                print("lr decay step #2")
                for param_group in optim.param_groups:
                    param_group['lr'] = current_lr
                mlperf_log.ssd_print(key=mlperf_log.OPT_LR,
                                     value=current_lr)
            if use_cuda:
                img = img.cuda()
            img = Variable(img, requires_grad=True)
            ploc, plabel = ssd300(img)
            trans_bbox = bbox.transpose(1,2).contiguous()
            if use_cuda:
                trans_bbox = trans_bbox.cuda()
                label = label.cuda()
            gloc, glabel = Variable(trans_bbox, requires_grad=False), \
                           Variable(label, requires_grad=False)
            loss = loss_func(ploc, plabel, gloc, glabel)

            if not np.isinf(loss.item()): avg_loss = 0.999*avg_loss + 0.001*loss.item()

            print("Iteration: {:6d}, Loss function: {:5.3f}, Average Loss: {:.3f}"\
                        .format(iter_num, loss.item(), avg_loss), end="\r")
            optim.zero_grad()
            loss.backward()
            optim.step()

            if iter_num in args.evaluation:
                if not args.no_save:
                    print("")
                    print("saving model...")
                    torch.save({"model" : ssd300.state_dict(), "label_map": train_coco.label_info},
                                "./models/iter_{}.pt".format(iter_num))

                mlperf_log.ssd_print(key=mlperf_log.EVAL_START, value=iter_num)
                if coco_eval(ssd300, val_coco, cocoGt, encoder, inv_map,
                             args.threshold, iter_num):
                    return True
                mlperf_log.ssd_print(key=mlperf_log.EVAL_STOP, value=iter_num)

            iter_num += 1
    return False

def main():
    args = parse_args()

    if not os.path.isdir('./models'):
        os.mkdir('./models')

    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed)
        np.random.seed(seed=args.seed)

    torch.backends.cudnn.benchmark = True

    # start timing here
    mlperf_log.ssd_print(key=mlperf_log.RUN_START)
    # check eval target
    mlperf_log.ssd_print(key=mlperf_log.EVAL_TARGET, value=args.threshold)

    success = train300_mlperf_coco(args)

    # end timing here
    mlperf_log.ssd_print(key=mlperf_log.RUN_STOP, value={"success": success})
    mlperf_log.ssd_print(key=mlperf_log.RUN_FINAL)

if __name__ == "__main__":
    main()
