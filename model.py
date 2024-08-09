import torch
from torch import nn
from torch_geometric.nn import GAT

class GATLayer(nn.Module):
    def __init__(self,in_features,out_features,dropout,alpha,concat=True):
        super().__init__()
        self.in_features=in_features # 节点向量的特征维度
        self.out_features=out_features # 经过GAT之后的特征维度
        self.dropout=dropout # droupout的参数
        self.alpha=alpha  # leakyReLU的参数

        # 定义可训练的 参数，即论文中的W和a
        self.W=nn.Parameter(torch.zeros(size=(in_features,out_features)))
        nn.init.xavier_normal_(self.W.data,gain=1.414) # xaiver初始化
        self.a=nn.Parameter(torch.zeros(size=(2*out_features,1)))
        nn.init.xavier_normal_(self.a.data,gain=1.414) # xaiver初始化

        # 定义leakyReLU激活函数
        self.leakpurelu=nn.LeakyReLU(self.alpha)

    def forward(self,input_h,adj):
        '''
        :param input_h: [N,in_features]，输入是N个节点，每个节点的维度是in_features
        :param adj: 图的邻接矩阵，维度[N,N]非零即一,N个节点
        :return:
        input_h:[N,in_features]，输入是N个节点，每个节点的维度是in_features
        W=(in_features,out_features)
        h=torch.mm(input_h,self.W)  [N,out_features]
        '''
        h=torch.mm(input_h,self.W) # 用共享参数W对每个节点的向量h进行线性变换,[N,out_features]

        N=h.size()[0] # 图的节点数
        input_concat=torch.cat([h.repeat(1,N).view(N*N,-1),h.repeat(N,1)],dim=1).view(N,-1,2*self.out_features)


