import torch
import torch.nn as nn

class SegModel(nn.Module):
    def __init__(self, pretrained_model):
        super(SegModel, self).__init__()
        # Load the pretrained ResNet18 model
        self.pretrained_model = pretrained_model
        self.upsample5 = nn.Upsample(scale_factor=5)
        self.upsample4 = nn.Upsample(scale_factor=5)
        self.upsample3 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample1 = nn.Upsample(scale_factor=2)

        self.conv5 = nn.Conv1d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()

        self.conv4 = nn.Conv1d(1024, 256, kernel_size=5, stride=1, padding=2)
        self.relu4 = nn.ReLU()

        self.conv3 = nn.Conv1d(512, 128, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU()

        self.conv2 = nn.Conv1d(256, 64, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()

        self.conv_out1 = nn.Conv1d(128, 32, kernel_size=5, stride=1, padding=2)
        self.conv_out2 = nn.Conv1d(32, 7, kernel_size=5, stride=1, padding=2)
        self.relu_out = nn.ReLU()
        # Define additional layers
        # self.additional_conv = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        # self.additional_fc = nn.Linear(256 * 7 * 7, 1000)  # Adjust the input size based on your needs

    def forward(self, x):
        # Get intermediate feature maps from the pretrained ResNet18
        feature_extractor = self.pretrained_model.feature_extractor
        layer1 = self.get_layer_internal_result(feature_extractor.layer1, x)
        layer2 = self.get_layer_internal_result(feature_extractor.layer2, layer1[-1])
        layer3 = self.get_layer_internal_result(feature_extractor.layer3, layer2[-1])
        layer4 = self.get_layer_internal_result(feature_extractor.layer4, layer3[-1])
        layer5 = self.get_layer_internal_result(feature_extractor.layer5, layer4[-1])
        
        skip_con5 = self.upsample5(layer5[-2])  # 1024X15
        conv5 = self.conv5(skip_con5)           # 512X15
        conv5 = self.relu5(conv5)
        concat4 = torch.cat([layer4[-2], conv5], dim=1) # 1024X15

        skip_con4 = self.upsample4(concat4)     # 1024X75 
        conv4 = self.conv4(skip_con4)           # 256X75    
        conv4 = self.relu4(conv4)
        concat3 = torch.cat([layer3[-2], conv4], dim=1) # 512X75

        skip_con3 = self.upsample3(concat3)     # 512X150
        conv3 = self.conv3(skip_con3)           # 128X150    
        conv3 = self.relu3(conv3)
        concat2 = torch.cat([layer2[-2], conv3], dim=1) # 256X150

        skip_con2 = self.upsample2(concat2)     # 256X300
        conv2 = self.conv2(skip_con2)           # 64X300    
        conv2 = self.relu2(conv2)
        concat1 = torch.cat([layer1[-2], conv2], dim=1) # 128X300

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