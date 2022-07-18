# model_unet.py
import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# U-net model with 1D convolutions
class unet_model(nn.Module):
    """
    U-net model with 1D convolutions
    """
    def __init__(self,
                 conv_input_dim=1,
                 conv_kernel=[3,3,3],
                 conv_feature=[16,32,64],
                 output_dim=1):
        """
        Model definition

        Parameters
        ----------
        conv_input_dim : number of features per ECG signal, in input
            int. The default is 1.
        conv_kernel : list of convolution kernels for encoder and decoder
            list of int. The default is [3,3,3].
        conv_feature : list of convolution features, len(conv_feature) = len(conv_kernel)
            list of int. The default is [16,32,64].
        output_dim : number of features per ECG signal, in output.
            int. The default is 1.

        """
        # Inherit everything from the nn.Module
        super().__init__()
        
        # Define layers
        self.encoder = nn.ModuleList()
        in_features = conv_input_dim
        
        # Encoder
        for kernel,features in zip(conv_kernel,conv_feature):
            self.encoder.append(ConvBlock(kernel, in_features, features, decoder=False))
            in_features = features
        
        # Bridge
        self.bridge = ConvBlock(kernel, features, 2*features, decoder=False)
        
        # Decoder
        in_features = 2*features
        self.decoder = nn.ModuleList()
        for kernel,features in zip(reversed(conv_kernel),reversed(conv_feature)):
            self.decoder.append(ConvBlock(kernel, in_features, features, decoder=True))
            in_features = features
        
        # max pooling
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Last layer
        self.last = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=output_dim, kernel_size=conv_kernel[-1], padding='same'),
                                  nn.Sigmoid())      
    
    def forward(self, x, hidden=None):
        """
        Forward function

        Parameters
        ----------
        x : Tensor of ECG signal
            size [batch, seq, signal_length].
        hidden : Tensor of short-long term memory
            size [n_layers*n_directions, batch, hidden]. The default is None.
            
        Returns
        -------
        y_pred : Tensor of peaks in ECG signal, range [0,1]
            size [batch, seq, signal_length].

        """
        # length of sequeence, and permute order
        self.seq_length = x.size(1) 
        x = torch.permute(x, (0, 2, 1)) # from [batch, seq_len, 1] to [batch, 1, seq_len]
        
        # encoder
        x_history = []
        for conv in self.encoder:
            x = conv(x)
            x_history.append(x)
            x = self.maxpool(x)
            
        # bridge and decoder
        x = self.bridge(x)
        for upconv in self.decoder:
            x = upconv(x,x_history.pop())
        
        # last layer and permute back the dimensions
        cnn_out = self.last(x)
        y_pred = torch.permute(cnn_out, (0, 2, 1)) # from [batch, 1, seq_len] to [batch, seq_len, 1]
        
        return y_pred

# U-net model with Res-Inception blocks of 1D convolutions
class unet_inception_model(nn.Module):
    """
    U-net model with Res-Inception 1D convolutions blocks
    """
    def __init__(self,
                 conv_input_dim=1,
                 conv_kernel_res=[15,17,19,21],
                 conv_feature=[32,64,128,256],
                 stride_size=[2,2,2,5],
                 output_dim=1):
        """
        Model definition

        Parameters
        ----------
        conv_input_dim : number of features per ECG signal, in input
            int. The default is 1.
        conv_kernel_res : list of convolution kernels applied in every block,
            different resolutions applied in every block.
            list of int. The default is [15,17,19,21].
        conv_feature : TYPE, optional
            list of int. The default is [32,64,128,256].
        stride_size : stride applied for every resolution kernel in res-inception block
            list of int. The default is [2,2,2,5].
        output_dim : number of features per ECG signal, in output.
            int. The default is 1.

        """
        # Inherit everything from the nn.Module
        super().__init__()        
        # Encoder
        features_inc = [int(conv_feature[0]/4)]*4
        self.encode1 = nn.Sequential(
                                nn.Conv1d(conv_input_dim, conv_feature[0], kernel_size=3, padding='same'),
                                nn.BatchNorm1d(num_features=conv_feature[0], affine=False),
                                nn.LeakyReLU(0.2,),
                                InceptionResBlock(conv_feature[0], kernels=conv_kernel_res, out_features=features_inc, stride_size=stride_size[0]))
        
        features_inc = [int(conv_feature[1]/4)]*4
        self.encode2 = nn.Sequential(
                                nn.LeakyReLU(0.2,),
                                nn.Conv1d(conv_feature[0], conv_feature[1], kernel_size=3, padding='same'),
                                nn.BatchNorm1d(num_features=conv_feature[1], affine=False),
                                InceptionResBlock(conv_feature[1], kernels=conv_kernel_res, out_features=features_inc, stride_size=stride_size[1]))
        
        features_inc = [int(conv_feature[2]/4)]*4
        self.encode3 = nn.Sequential(
                                nn.LeakyReLU(0.2,),
                                nn.Conv1d(conv_feature[1], conv_feature[2], kernel_size=3, padding='same'),
                                nn.BatchNorm1d(num_features=conv_feature[2], affine=False),
                                InceptionResBlock(conv_feature[2], kernels=conv_kernel_res, out_features=features_inc, stride_size=stride_size[2]))
        
        features_inc = [int(conv_feature[3]/4)]*4
        self.encode4 = nn.Sequential(
                                nn.LeakyReLU(0.2,),
                                nn.Conv1d(conv_feature[2], conv_feature[3], kernel_size=3, padding='same'),
                                nn.BatchNorm1d(num_features=conv_feature[3], affine=False),
                                InceptionResBlock(conv_feature[3], kernels=conv_kernel_res, out_features=features_inc, stride_size=stride_size[3]))
        # create encoder as module list
        self.encoder = [self.encode1,self.encode2,self.encode3,self.encode4]
        
        # Bridge
        input_to_bridge = sum(features_inc)
        self.bridge = InceptionResBlock(input_to_bridge, kernels=conv_kernel_res, out_features=features_inc, stride_size=1)
        
        features_inc = [int(conv_feature[2]/4)]*4
        self.decoder1 = nn.Sequential(
                                nn.LeakyReLU(0.2,),
                                nn.ConvTranspose1d(2*conv_feature[3], conv_feature[2], kernel_size=5, padding=0, stride=5),
                                nn.BatchNorm1d(conv_feature[2]),
                                InceptionResBlock(conv_feature[2], kernels=conv_kernel_res, out_features=features_inc, stride_size=1))
        
        features_inc = [int(conv_feature[1]/4)]*4
        self.decoder2 = nn.Sequential(
                                nn.LeakyReLU(0.2,),
                                nn.ConvTranspose1d(2*conv_feature[2], conv_feature[1], kernel_size=4, padding=1, stride=2),
                                nn.BatchNorm1d(conv_feature[1]),
                                InceptionResBlock(conv_feature[1], kernels=conv_kernel_res, out_features=features_inc, stride_size=1))
        
        features_inc = [int(conv_feature[0]/4)]*4
        self.decoder3 = nn.Sequential(
                                nn.LeakyReLU(0.2,),
                                nn.ConvTranspose1d(2*conv_feature[1], conv_feature[0], kernel_size=4, padding=1, stride=2),
                                nn.BatchNorm1d(conv_feature[0]),
                                InceptionResBlock(conv_feature[0], kernels=conv_kernel_res, out_features=features_inc, stride_size=1))
        
        features_inc = [int(conv_feature[0]/4)]*4
        self.decoder4 = nn.Sequential(
                                nn.LeakyReLU(0.2,),
                                nn.ConvTranspose1d(2*conv_feature[0], conv_feature[0], kernel_size=4, padding=1, stride=2),
                                nn.BatchNorm1d(conv_feature[0]),
                                InceptionResBlock(conv_feature[0], kernels=conv_kernel_res, out_features=features_inc, stride_size=1))
        # create encoder as module list
        self.decoder = [self.decoder1,self.decoder2,self.decoder3,self.decoder4]
        
        # Last layers
        self.last = nn.Sequential(nn.Conv1d(in_channels=conv_feature[0], out_channels=output_dim, kernel_size=3, padding='same'),
                                  nn.Sigmoid())      
    
    def forward(self, x, hidden=None):
        """
        Forward function

        Parameters
        ----------
        x : Tensor of ECG signal
            size [batch_size, sequence, signal_length, conv_input_dim].
        hidden : Tensor of short-long term memory
            size [n_layers*n_directions, batch, hidden]. The default is None.
            
        Returns
        -------
        y_pred : Tensor of peaks in ECG signal, range [0,1]
            size [batch_size, sequence, signal_length, output_dim].

        """
        # length of sequeence, and permute order
        self.seq_length = x.size(1) 
        x = torch.permute(x, (0, 2, 1)) # from [batch, seq_len, 1] to [batch, 1, seq_len]
        
        # encoder
        x_history = []
        for conv in self.encoder:
            x = conv(x)
            x_history.append(x)
                    
        # bridge and decoder
        x = self.bridge(x)
        for upconv in self.decoder:
            x = torch.cat((x,x_history.pop()), dim=1)
            x = upconv(x)
        
        # last layer and permute back the dimensions
        cnn_out = self.last(x)
        y_pred = torch.permute(cnn_out, (0, 2, 1)) # from [batch, 1, seq_len] to [batch, seq_len, 1]
        
        return y_pred
    

#%% Helper functions
class ConvBlock(nn.Module):
    """
    Two 1D convolutions block, followed by batch normalization and leakyRelu.
    Convtranspose can be added before the convulotions, used for U-net architecture.
    """

    def __init__(self, 
                 kernel, 
                 in_features, 
                 out_features, 
                 decoder=False):
        """
        Block definition

        Parameters
        ----------
        kernel : kernel size
            int.
        in_features : input number of features
            int.
        out_features : output number of features
            int.
        decoder : add a Convtranspose layer before convulotions or not?
            boolean. The default is False.

        """
        super().__init__()
        
        # if we first do upconv
        self.decoder = decoder
        if self.decoder:
            self.upconv = nn.ConvTranspose1d(in_channels=in_features, out_channels=out_features, stride=2, kernel_size=4, padding=1)
            self.batch_d = nn.BatchNorm1d(num_features=out_features, affine=False)
            self.activation1_d = nn.LeakyReLU(0.2)
            
        # First CNN 
        self.conv1 = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=kernel, padding='same')
        self.batch1 = nn.BatchNorm1d(num_features=out_features, affine=False)
        self.activation1 = nn.LeakyReLU(0.2)
        
        # Second CNN
        self.conv2 = nn.Conv1d(in_channels=out_features, out_channels=out_features, kernel_size=kernel, padding='same')
        self.batch2 = nn.BatchNorm1d(num_features=out_features, affine=False)
        self.activation2 = nn.LeakyReLU(0.2)

    def forward(self, x, x_past=None):
        """
        Forward function

        Parameters
        ----------
        x : intput Tensor
            size [batch, in_features, tensor_length].
        x_past : intput Tensor to decoder, from symetric encoder layer 
            size [batch, in_features, tensor_length, ]. The default is None.

        Returns
        -------
        out : output Tensor
            size [batch, out_features, tensor_length].

        """
        # if we increase upconv
        if self.decoder:
            x = self.activation1_d(self.batch_d(self.upconv(x)))
            x = torch.cat((x,x_past),dim=1)
            
        # rest of CNN
        x = self.activation1(self.batch1(self.conv1(x)))
        out = self.activation2(self.batch2(self.conv2(x)))
            
        return out
    
class InceptionResBlock(nn.Module):
    """
    Residual Inception Blcok. 
    Inception - process in input tensor in multiple resolutions, in every 
    resolution, first decrease the number of feature with kernel of 1, than
    and than a second convolution for a desired number of features. 
    Residual - add the input signal to the output signal (after concat features
    in different resolutions), so the unit transform can be learned. 
    """
    def __init__(self, 
                 in_features, 
                 kernels=[15,17,19,21], 
                 out_features=[16,16,16,16], 
                 stride_size=1):
        '''
        Block definition

        Parameters
        ----------
        in_features : number of features of input signal 
            int.
        kernels : size of kernel for every resolution
            list of int. The default is [15,17,19,21].
        out_features : number of features for every resolution  
            list of int. The default is [16,16,16,16].
        stride_size : size of stride in every resolution, 
            len(stride_size) = len(out_features) = len(kernels)
            list of int. The default is 1.

        '''
        super().__init__()
        # define params
        padding = [int((kernel-1)/2) for kernel in kernels]
        total_feature = sum(out_features)
        
        # define res conv layers
        self.res = nn.Conv1d(in_channels=in_features, out_channels=total_feature, kernel_size=1, stride=stride_size)
        
        # define first branch
        self.conv01 = nn.Conv1d(in_channels=in_features, out_channels=out_features[0], kernel_size=1, stride=stride_size)
        self.conv1 = nn.Conv1d(in_channels=out_features[0], out_channels=out_features[0], kernel_size=kernels[0], padding=padding[0])
        
        # second branch
        self.conv02 = nn.Conv1d(in_channels=in_features, out_channels=out_features[1], kernel_size=1, stride=stride_size)
        self.conv2 = nn.Conv1d(in_channels=out_features[1], out_channels=out_features[1], kernel_size=kernels[1], padding=padding[1])
        
        # thrid branch
        self.conv03 = nn.Conv1d(in_channels=in_features, out_channels=out_features[2], kernel_size=1, stride=stride_size)
        self.conv3 = nn.Conv1d(in_channels=out_features[2], out_channels=out_features[2], kernel_size=kernels[2], padding=padding[2])
        
        # fourth branch
        self.conv04 = nn.Conv1d(in_channels=in_features, out_channels=out_features[3], kernel_size=1, stride=stride_size)
        self.conv4 = nn.Conv1d(in_channels=out_features[3], out_channels=out_features[3], kernel_size=kernels[3], padding=padding[3])
        
        # additional layers
        self.bn_res = nn.BatchNorm1d(num_features=total_feature, affine=False)
        self.bn0 = nn.BatchNorm1d(num_features=out_features[0], affine=False)
        self.bn1 = nn.BatchNorm1d(num_features=out_features[1], affine=False)
        self.bn2 = nn.BatchNorm1d(num_features=out_features[2], affine=False)
        self.bn3 = nn.BatchNorm1d(num_features=out_features[3], affine=False)
        self.lrelu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        """
        Forward function

        Parameters
        ----------
        x : input Tensor
            size [batch, in_features, tensor_length].

        Returns
        -------
        out : output Tensor
            size [batch, sum(out_features), tensor_length].

        """
        # run conv for every branch in Inception Res block
        x_res = self.bn_res(self.res(x))
        x1 = self.bn0(self.conv1(self.lrelu(self.bn0(self.conv01(x)))))
        x2 = self.bn1(self.conv2(self.lrelu(self.bn1(self.conv02(x)))))
        x3 = self.bn2(self.conv3(self.lrelu(self.bn2(self.conv03(x)))))
        x4 = self.bn3(self.conv4(self.lrelu(self.bn3(self.conv04(x)))))
        
        # concat and res
        out = torch.cat((x1,x2,x3,x4), dim=1)
        out += x_res
        return out
    

