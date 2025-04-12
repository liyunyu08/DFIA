import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone_res12 import ResNet
# from models.resnet10 import ResNet
from models.conv4 import ConvNet
# from models.conv4_V2 import ConvNet
# from models.transformer import Vit
# from high_pass import High_Passr


class DFIA(nn.Module):

    def __init__(self, args, resnet=False, mode=None):
        super().__init__()
        self.mode = mode
        self.args = args
        self.resnet = resnet
        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.method  = args.method
        # self.vit = Vit(64, 2,1)
        # self.coff = args.coff
        # self.high_pass =High_Pass()

        self.k = args.way * args.shot

        if resnet:
            self.encoder = ResNet()
            self.fc = nn.Linear(512, self.args.num_class)
            print("This is ResNet")

        else:
            self.encoder = ConvNet()
            self.fc = nn.Linear(64, self.args.num_class)
            print("This is ConvNet")




    def forward(self, input,Re=True):
        if self.mode == 'fc':
            return self.fc_forward(input)

        elif self.mode == 'encoder':

            x1,x2 = self.encoder(input,Re)

            return x1,x2

        # elif self.mode == 'high_pass':
        #     image =input
        #     x =self.high_pass(image,self.coff)
        #
        #     return x


        # elif self.mode == 'vit':
        #     spa_feat, freq_feat =input
        #     x =self.fuseblock(spa_feat,freq_feat)
        #     return x


        elif self.mode == 'base':
            spt,qry = input

            return self.metric(spt,qry)

        else:
            raise ValueError('Unknown mode')


    def fc_forward(self, x):
        x = x.mean(dim=[-1,-2])
        return self.fc(x)



    def metric(self, token_support, token_query):
        qry_pooled = token_query.mean(dim=[-1,-2])

        token_spt = self.normalize_feature(token_support)
        token_qry = self.normalize_feature(token_query)

        way = token_spt.shape[0]
        num_qry = token_qry.shape[0]

        token_spt = token_spt.unsqueeze(0).repeat(num_qry, 1, 1, 1, 1)
        token_qry = token_qry.unsqueeze(1).repeat(1, way, 1, 1, 1)

        spt_attended_pooled = token_spt.mean(dim=[-1,-2])
        qry_attended_pooled = token_qry.mean(dim=[-1,-2])

        similarity_matrix = F.cosine_similarity(spt_attended_pooled, qry_attended_pooled, dim=-1)
        logits = similarity_matrix * self.scale

        if self.training:
            return logits, self.fc(qry_pooled)
        else:
            return logits



    def normalize_feature(self, x):
        return x - x.mean(1).unsqueeze(1)


