from datasets.download import streaming_download_manager
from torchvision.models import vgg19, VGG19_Weights
import torch.nn as nn

class VGGLoss(nn.Module):
    def __init__(self, device:streaming_download_manager):
        super().__init__()
        #This will get by default up to the last Convolution before
        #the MaxPooling layer, the fifth one, that happens to be the 
        #one used to compute the vgg loss for the best performing 
        #model in the paper
        self.vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:36].\
                                                          eval().to(device) 
        for param in self.vgg.parameters():
            param.requires_grad=False
        self.mse = nn.MSELoss()

    def forward(self, gen, gt):
        gen_vgg = self.vgg(gen)
        gt_vgg  = self.vgg(gt)
        return self.mse(gen_vgg, gt_vgg)
        