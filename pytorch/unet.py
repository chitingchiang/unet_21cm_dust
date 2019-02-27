import torch
import torch.nn as nn
import time

class UNet(nn.Module):
    def __init__(self,n_channel_in=1,n_channel_first=64,n_channel_out=1,n_depth=4,n_convpatch=3,lrelu_slope=0.,dp_rate=0.5):
        super(UNet,self).__init__()

        self.n_depth = n_depth
        self.lrelu_slope = lrelu_slope

        n_convpad = (n_convpatch-1)//2

        self.in_conv = DoubleConv(n_channel_in,n_channel_first,n_convpatch,n_convpad,lrelu_slope,dp_rate)

        n_ch_in = n_channel_first
        n_ch_out = n_ch_in*2

        self.down_convs = []
        for i in range(n_depth-1):
            self.down_convs.append(DownConv(n_ch_in,n_ch_out,n_convpatch,n_convpad,lrelu_slope,dp_rate))
            n_ch_in = n_ch_in*2
            n_ch_out = n_ch_out*2
        n_ch_out = n_ch_in
        self.down_convs.append(DownConv(n_ch_in,n_ch_out,n_convpatch,n_convpad,lrelu_slope,dp_rate))

        self.down_convs = nn.ModuleList(self.down_convs)

        n_ch_out = n_ch_in//2

        self.up_convs = []
        for i in range(n_depth-1):
            self.up_convs.append(UpConv(n_ch_in,n_ch_out,n_convpatch,n_convpad,lrelu_slope,dp_rate))
            n_ch_in = n_ch_in//2
            n_ch_out = n_ch_out//2
        n_ch_out = n_ch_in
        self.up_convs.append(UpConv(n_ch_in,n_ch_out,n_convpatch,n_convpad,lrelu_slope,dp_rate))
        self.up_convs = nn.ModuleList(self.up_convs)

        self.final_conv = nn.Conv2d(in_channels=n_ch_in,out_channels=n_channel_out,kernel_size=1,padding=0)

    def forward(self,x):
        x = self.in_conv(x)

        x_down = []
        for i in range(self.n_depth):
            x_down.append(x)
            x = self.down_convs[i](x)

        for i in range(self.n_depth):
            x = self.up_convs[i](x,x_down[-i-1])

        x = self.final_conv(x)

        return x

class DoubleConv(nn.Module):
    def __init__(self,n_ch_in,n_ch_out,n_convpatch,n_convpad,lrelu_slope,dp_rate):
        super(DoubleConv,self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=n_ch_in,out_channels=n_ch_out,kernel_size=n_convpatch,padding=n_convpad,bias=False),
            nn.BatchNorm2d(num_features=n_ch_out),
            nn.LeakyReLU(negative_slope=lrelu_slope),
            nn.Dropout2d(p=dp_rate),
            nn.Conv2d(in_channels=n_ch_out,out_channels=n_ch_out,kernel_size=n_convpatch,padding=n_convpad,bias=False),
            nn.BatchNorm2d(num_features=n_ch_out),
            nn.LeakyReLU(negative_slope=lrelu_slope),
            nn.Dropout2d(p=dp_rate)
        )

    def forward(self,x):
        x = self.double_conv(x)
        return x

class DownConv(nn.Module):
    def __init__(self,n_ch_in,n_ch_out,n_convpatch,n_convpad,lrelu_slope,dp_rate):
        super(DownConv,self).__init__()
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2),
            DoubleConv(n_ch_in,n_ch_out,n_convpatch,n_convpad,lrelu_slope,dp_rate)
        )

    def forward(self,x):
        x = self.down_conv(x)
        return x

class UpConv(nn.Module):
    def __init__(self,n_ch_in,n_ch_out,n_convpatch,n_convpad,lrelu_slope,dp_rate):
        super(UpConv,self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels=n_ch_in,out_channels=n_ch_in,kernel_size=2,stride=2)
        self.double_conv = DoubleConv(n_ch_in*2,n_ch_out,n_convpatch,n_convpad,lrelu_slope,dp_rate)

    def forward(self,x1,x2):
        x = self.up_conv(x1)
        x = torch.cat((x,x2),dim=1)
        x = self.double_conv(x)
        return x

if __name__ == "__main__":
    """
    testing
    """
    n_batch = 32
    n_pixel = 64
    n_channel_in = 50
    n_channel_first = 64
    n_channel_out = 1
    n_depth = 4
    n_convpatch = 3

    image = torch.rand(n_batch,n_channel_in,n_pixel,n_pixel)
    y = torch.rand(n_batch,1,n_pixel,n_pixel)

    use_cuda = True
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    model = UNet(n_channel_in,n_channel_first,n_channel_out,n_depth,n_convpatch,0.).to(device)
    print(model)

    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    print(pp)

    time1 = time.time()

    image = image.to(device)
    y = y.to(device)

    y_pred = model(image)
    print(image.size())
    print(y_pred.size())
    loss = torch.sum((y-y_pred)**2)/n_batch
    print(loss)
    loss.backward()

    time2 = time.time()

    print(time2-time1)
