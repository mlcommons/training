import torch
import torch.nn as nn
from base_model import VGG16, L2Norm

class SSD300(nn.Module):
    """
        Build a SSD module to take 300x300 image input,
        and output 8732 per class bounding boxes

        vggt: pretrained vgg16 (partial) model
        label_num: number of classes (including background 0)
    """
    def __init__(self, label_num, vgg_path="./vgg16n.pth"):

        super(SSD300, self).__init__()

        self.label_num = label_num

        self.l2norm4 = L2Norm()

        self.vggt = VGG16([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', \
                           512, 512, 512, 'M' , 512, 512, 512])
        #print(torch.sum(list(self.vggt.parameters())[0]))
        #print(self.vggt.state_dict().keys())
        if vgg_path is not None:
            print("loading pretrained vgg model", vgg_path)
            self.vggt.load_state_dict(torch.load(vgg_path))
        #print(torch.sum(list(self.vggt.parameters())[0]))


        # conv8_1, conv8_2
        self.block8 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
        )

        # conv9_1, conv9_2
        self.block9 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
        )

        # conv10_1, conv10_2
        self.block10 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        # conv11_1, conv11_2
        self.block11 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        # after l2norm, conv7, conv8_2, conv9_2, conv10_2, conv11_2
        # classifer 1, 2, 3, 4, 5 ,6

        self.num_defaults = [4, 6, 6, 6, 4, 4]
        self.out_chan = [512, 1024, 512, 256, 256, 256]
        self.loc = []
        self.conf = []

        for nd, oc in zip(self.num_defaults, self.out_chan):
            self.loc.append(nn.Conv2d(oc, nd*4, kernel_size=3, padding=1))
            self.conf.append(nn.Conv2d(oc, nd*label_num, kernel_size=3, padding=1))
        
        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)
        # intitalize all weights
        self._init_weights()
    
    def _init_weights(self):
        
        layers = [
            self.block8, self.block9, self.block10, self.block11,
            *self.loc, *self.conf]

        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1: nn.init.xavier_uniform_(param)

    # Shape the classifier to the view of bboxes
    def bbox_view(self, src, loc, conf):
        ret = []
        for s, l, c in zip(src, loc, conf):
            ret.append((l(s).view(s.size(0), 4, -1), c(s).view(s.size(0), self.label_num, -1)))

        locs, confs = list(zip(*ret))
        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        return locs, confs
        #locs, confs = torch.cat(locs, 1).contiguous(), torch.cat(confs, 1).contiguous()
        #return locs.view(locs.size(0), 4, -1).contiguous(), \
        #       confs.view(confs.size(0), self.label_num, -1).contiguous()

    def forward(self, data):

        layer4, layer7 = self.vggt(data)
        #print(layer4.shape, layer5.shape)
        #print(layer6.shape)
        layer8  = self.block8(layer7)
        layer9  = self.block9(layer8)
        layer10 = self.block10(layer9)
        #print(layer10.shape)
        layer11 = self.block11(layer10)

        src = [self.l2norm4(layer4), layer7, layer8, layer9, layer10, layer11]
        # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4

        locs, confs = self.bbox_view(src, self.loc, self.conf)

        # For SSD 300, shall return nbatch x 8732 x {nlabels, nlocs} results
        return locs, confs
