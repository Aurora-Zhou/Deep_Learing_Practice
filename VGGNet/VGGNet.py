import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()   #初始化父类的类型
        self.feature = features  # 用于提取图像的特征
        self.classifier = nn.Sequential(  # 一个连续的容器。模块将按照在构造函数中传递的顺序添加到模块中。
            nn.Dropout(p=0.5),  # 随机失活一部分神经元，用于减少过拟合，默认比例为0.5，仅用于正向传播。
            nn.Linear(512 * 7 * 7, 4096),  # 全连接   512*7*7展平之后得到的一维向量的个数，4096是全连接层的结点个数。
            nn.ReLU(True),  # relu函数：f(x) = Max (0 ,x)
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),  # 定义第二个全连接层，输入为4096，本层的结点数为4096
            nn.ReLU(True),
            nn.Linear(4096, num_classes)  # 定义最后一个全连接层。num_classes：分类的类别个数。
        )
        if init_weights:
            self._initialize_weights()

    def forward(self,x):   #定义前向传播过程   x为输入的图像数据(x代表输入的数据)
        # N x 3 x 224 x 224
        x = self.features(x)   #将x输入到features，并赋值给x
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)    #进行展平处理，start_dim=1，指定从哪个维度开始展平处理，因为第一个维度是batch，不需要对它进行展开，所以从第二个维度进行展开。
        # N x 512*7*7
        x = self.classifier(x)     #展平后将特征矩阵输入到事先定义好的分类网络结构中。
        return x

    def _initialize_weights(self):
        for m in self.modules():    #用m遍历网络的每一个子模块，即网络的每一层
            if isinstance(m, nn.Conv2d):    #若m的值为 nn.Conv2d,即卷积层 Conv2d：对由多个输入平面组成的输入信号应用2D卷积。
                # nn.init.kaiming_normal_(m.weight, mode=‘fan_out’, nonlinearity=‘relu’)   #凯明初始化方法
                nn.init.xavier_uniform_(m.weight)   #用xavier初始化方法初始化卷积核的权重
                if m.bias is not None:     #若偏置不为None，则将偏置初始化为0
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):   #若m的值为nn.Linear,即池化层
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)    #用一个正太分布来给权重进行赋值，0为均值，0.01为方差
                nn.init.constant_(m.bias, 0)    #对权重进行赋值，并且将偏置初始化为0