import torch
import torch.nn as nn
WIN_SIZE=300
import ipdb

class SegModel(nn.Module):
    def __init__(self, pretrained_model,multi_windows=True):
        super(SegModel, self).__init__()
        # Load the pretrained ResNet18 model
        self.multi_windows=multi_windows
        self.pretrained_model = pretrained_model
        if multi_windows:
            factor = 3
        else:
            factor = 1
        self.factor = factor
        self.upsample5 = nn.Upsample(scale_factor=5)
        self.upsample4 = nn.Upsample(scale_factor=5)
        self.upsample3 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample1 = nn.Upsample(scale_factor=2)

        self.conv5 = nn.Conv1d(factor*1024, 512, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()

        self.conv4 = nn.Conv1d((factor+1)*512, 256, kernel_size=5, stride=1, padding=2)
        self.relu4 = nn.ReLU()

        self.conv3 = nn.Conv1d((factor+1)*256, 128, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU()

        self.conv2 = nn.Conv1d((factor+1)*128, 64, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        
        self.conv_out1 = nn.Conv1d((factor+1)*64, 32, kernel_size=5, stride=1, padding=2)
        #self.conv_out2 = nn.Conv1d((factor+1)*16, 7, kernel_size=5, stride=1, padding=2)
        self.conv_out2 = nn.Conv1d(32, 7, kernel_size=5, stride=1, padding=2)
        self.relu_out = nn.ReLU()

        # Define additional layers
        # self.additional_conv = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        # self.additional_fc = nn.Linear(256 * 7 * 7, 1000)  # Adjust the input size based on your needs

    def forward(self, x):
        # Get intermediate feature maps from the pretrained ResNet18
        feature_extractor = self.pretrained_model.feature_extractor
        layer1_list = []
        layer2_list = []
        layer3_list = []
        layer4_list = []
        layer5_list = []
        for i in range(self.factor):
            #layer1_list.append(self.get_layer_internal_result(feature_extractor.layer1, x[i*WIN_SIZE:(i+1)*WIN_SIZE, :, :]))
            layer1_list.append(self.get_layer_internal_result(feature_extractor.layer1, x[:,:,i*WIN_SIZE:(i+1)*WIN_SIZE]))
            layer2_list.append(self.get_layer_internal_result(feature_extractor.layer2, layer1_list[i][-1]))
            layer3_list.append(self.get_layer_internal_result(feature_extractor.layer3, layer2_list[i][-1]))
            layer4_list.append(self.get_layer_internal_result(feature_extractor.layer4, layer3_list[i][-1]))
            layer5_list.append(self.get_layer_internal_result(feature_extractor.layer5, layer4_list[i][-1]))
        layer5_out = torch.cat([out[-2] for out in layer5_list], dim=1)
        layer4_out = torch.cat([out[-2] for out in layer4_list], dim=1)
        layer3_out = torch.cat([out[-2] for out in layer3_list], dim=1)
        layer2_out = torch.cat([out[-2] for out in layer2_list], dim=1)
        layer1_out = torch.cat([out[-2] for out in layer1_list], dim=1)
        skip_con5 = self.upsample5(layer5_out)  # 1024X15
        conv5 = self.conv5(skip_con5)           # 512X15
        conv5 = self.relu5(conv5)
        concat4 = torch.cat([layer4_out, conv5], dim=1) # 1024X15

        skip_con4 = self.upsample4(concat4)     # 1024X75 
        conv4 = self.conv4(skip_con4)           # 256X75    
        conv4 = self.relu4(conv4)
        concat3 = torch.cat([layer3_out, conv4], dim=1) # 512X75

        skip_con3 = self.upsample3(concat3)     # 512X150
        conv3 = self.conv3(skip_con3)           # 128X150    
        conv3 = self.relu3(conv3)
        concat2 = torch.cat([layer2_out, conv3], dim=1) # 256X150

        skip_con2 = self.upsample2(concat2)     # 256X300
        conv2 = self.conv2(skip_con2)           # 64X300    
        conv2 = self.relu2(conv2)
        concat1 = torch.cat([layer1_out, conv2], dim=1) # 128X300
        out = self.conv_out1(concat1) #32X300
        out = self.relu_out(out)
        out = self.conv_out2(out) #7X300

        # out = self.pretrained_model.classifier(layer5[-1][...,0])
        # Use feature maps as input to additional layers
        # x = self.additional_conv(layer4)
        # x = x.view(x.size(0), -1)  # Flatten the tensor before passing to the fully connected layer
        # x = self.additional_fc(x)

        return out
    
    def get_layer_internal_result(self, layer, x):
        inter_res = [x]
        for child in layer.children():
            inter_res.append(child(inter_res[-1]))
        return inter_res