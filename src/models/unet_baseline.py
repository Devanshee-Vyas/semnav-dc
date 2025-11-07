import torch, torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(True)
        )
    def forward(self, x): return self.net(x)

class UNetSmall(nn.Module):
    def __init__(self, in_ch=5, out_ch=1):
        super().__init__()
        self.d1=DoubleConv(in_ch,32); self.p1=nn.MaxPool2d(2)
        self.d2=DoubleConv(32,64);    self.p2=nn.MaxPool2d(2)
        self.b =DoubleConv(64,128)
        self.u2=nn.ConvTranspose2d(128,64,2,stride=2)
        self.d3=DoubleConv(128,64)
        self.u1=nn.ConvTranspose2d(64,32,2,stride=2)
        self.d4=DoubleConv(64,32)
        self.out=nn.Conv2d(32,out_ch,1)

    def forward(self,x):
        d1=self.d1(x); d2=self.d2(self.p1(d1)); b=self.b(self.p2(d2))
        u2=self.u2(b); d3=self.d3(torch.cat([u2,d2],1))
        u1=self.u1(d3); d4=self.d4(torch.cat([u1,d1],1))
        y=self.out(d4)
        return torch.relu(y)  # 深度 >= 0
