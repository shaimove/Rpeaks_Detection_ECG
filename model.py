# Models.py
import numpy as np
import torch
import torch.nn as nn
import math
from torch.autograd import Variable 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# First Model: Simple LSTM
class lstm_model(nn.Module):

    def __init__(self,input_dim, hidden_size, output_size, n_layers=1, dropout=0.2,
                 bidirectional=True):
        
        # Inherit everything from the nn.Module
        super().__init__()
        
        n_directions = 2 if bidirectional else 1
            
        # Define LSTM
        self.lstm = nn.LSTM(input_size = input_dim, hidden_size = hidden_size, num_layers = n_layers, 
                          dropout = (0 if n_layers == 1 else dropout_p), batch_first=True,
                          bidirectional = bidirectional) 
        
        # Define Linear 
        self.linear = nn.Sequential(
                        nn.Linear(hidden_size*n_directions, output_size),
                        nn.Sigmoid())
        
    def forward(self, input_seq, hidden=None):
        # Extract batch_size, Input: [batch, seq, hidden]
        self.seq_length = input_seq.size(1)
        
        # fisrt direction LSTM, hidden ([2*2, batch, hidden], [2*2, batch, hidden]), lstm_out [batch, seq, 2*hidden]
        lstm_out, hidden = self.lstm(input_seq, hidden)

        # linear and activation layer
        out = [self.linear(lstm_out[:,i,:]) for i in range(self.seq_length)]
        y_pred = torch.stack(out).T.permute(1,2,0)
        
        return y_pred
    
# Second model: CNN + LSTM
class cnn_lstm_model(nn.Module):

    def __init__(self,
                 conv_input_dim = 1,
                 conv_kernel = [3,3,3,3,3],
                 conv_feature = [16,32,32,16,1],
                 hidden_size_lstm = 64, 
                 n_layers_lstm = 1, 
                 dropout_lstm = 0.2,
                 bidirectional = True,
                 output_size = 1):
        
        # Inherit everything from the nn.Module
        super().__init__()
        
        # Define N layers CNN
        cnn_modules = []
        in_channels = conv_input_dim
        n_directions = 2
        
        for kernel,features in zip(conv_kernel,conv_feature):
            cnn_modules.append(nn.Conv1d(in_channels=in_channels, out_channels=features, kernel_size=kernel, padding='same'))
            cnn_modules.append(nn.BatchNorm1d(num_features = features, affine = False))
            cnn_modules.append(nn.ReLU())
            in_channels = features
        
        self.cnn = nn.Sequential(*cnn_modules)
        
        # Define LSTM
        self.lstm = nn.LSTM(input_size = conv_feature[-1], hidden_size = hidden_size_lstm, num_layers = n_layers_lstm, 
                          dropout = (0 if n_layers_lstm == 1 else dropout_p_lstm), batch_first=True,
                          bidirectional = bidirectional) 
        
        # Define Linear 
        self.linear = nn.Sequential(
                        nn.Linear(hidden_size_lstm*n_directions, output_size),
                        nn.Sigmoid())

        
    def forward(self, input_seq, hidden=None):
        # length of sequeence
        self.seq_length = input_seq.size(1) 
        
        # feature extraction: CNN
        input_seq = torch.permute(input_seq, (0, 2, 1)) # from [batch, seq_len, 1] to [batch, 1, seq_len]
        cnn_out = self.cnn(input_seq)
        cnn_out = torch.permute(cnn_out, (0, 2, 1)) # from [batch, 1, seq_len] to [batch, seq_len, 1]
        
        # fisrt direction LSTM, hidden ([2*2, batch, hidden], [2*2, batch, hidden]), lstm_out [batch, seq, 2*hidden]
        lstm_out, hidden = self.lstm(cnn_out, hidden)

        # linear and activation layer
        out = [self.linear(lstm_out[:,i,:]) for i in range(self.seq_length)]
        y_pred = torch.stack(out).T.permute(1,2,0)
        
        return y_pred
    
# Third model: Unet
class unet_model(nn.Module):

    def __init__(self,
                 conv_input_dim = 1,
                 conv_kernel = [3,3,3],
                 conv_feature = [16,32,64],
                 output_size = 1):
        
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
        self.last = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=output_size, kernel_size=conv_kernel[-1], padding='same'),
                                  nn.Sigmoid())      
    
    def forward(self, x, hidden=None):
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

# Third model: Unet
class unet_inception_model(nn.Module):

    def __init__(self,
                 conv_input_dim = 1,
                 conv_kernel_res = [15,17,19,21],
                 conv_feature = [32,64,128,256],
                 stride_size = [2,2,2,5],
                 output_size = 1):
        
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
        self.last = nn.Sequential(nn.Conv1d(in_channels=conv_feature[0], out_channels=output_size, kernel_size=3, padding='same'),
                                  nn.Sigmoid())      
    
    def forward(self, x, hidden=None):
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
    


    
    
# Forth Model: Simple Transformer
class transformer_model(nn.Module):

    def __init__(self, input_dim=1, cnn_kernel=3, num_features=128, num_of_attention_heads=8,
                 dim_feedforward=2048, num_of_encoder_leyrs=6, output_size=1, dropout_p=0.1, batch_first=True):
        
        # Inherit everything from the nn.Module
        super().__init__()
                
        # CNN as feature extraction 
        self.cnn = nn.Sequential(
                     nn.Conv1d(in_channels=input_dim, out_channels=num_features, kernel_size=cnn_kernel, padding='same'),
                     nn.BatchNorm1d(num_features=num_features, affine=False),
                     nn.ReLU(),
                     nn.Conv1d(in_channels=num_features, out_channels=num_features, kernel_size=cnn_kernel, padding='same'),
                     nn.BatchNorm1d(num_features=num_features, affine=False),
                     nn.ReLU())
        
        # Positional Encoding
        self.positional_encoder = PositionalEncoding(
                                    dim_model=num_features, dropout_p=dropout_p, max_len=1000)
        
        # Define Transfomer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model = num_features,
                                                        nhead = num_of_attention_heads,
                                                        dim_feedforward = dim_feedforward,
                                                        batch_first = batch_first)
        self.encoder_transformer = nn.TransformerEncoder(encoder_layer = self.encoder_layer,
                                                         num_layers = num_of_encoder_leyrs)
        
        # Define Linear 
        self.linear = nn.Sequential(
                        nn.Linear(num_features, output_size),
                        nn.Sigmoid())

    def forward(self, source):
        # T=S source and target sequence length, N batch size, E number of features 
        # source and target size is be (batch_size, sequence length)
        seq_len = source.shape[1]
        
        # permute before CNN, from [batch, seq_len, 1] to [batch, 1, seq_len]
        source = torch.permute(source, (0, 2, 1))
        # feature extraction
        source = self.cnn(source)
        # permute after cnn, from [batch, 1, seq_len] to [batch, seq_len, 1]
        source = torch.permute(source, (0, 2, 1))
        
        # positional encoding and get target mask
        source = self.positional_encoder(source) # [batch, seq, features]
        
        # transformer
        transformer_out = self.encoder_transformer(source) # [batch, seq, features]
        
        # linear and activation layer
        out = [self.linear(transformer_out[:,i,:]) for i in range(seq_len)]
        y_pred = torch.stack(out).T.permute(1,2,0)
        
        return y_pred

# Forth Model: Simple Transformer
class transformer_inception_model(nn.Module):

    def __init__(self, input_dim=1, conv_kernel_res=[15,17,19,21], num_features=128, num_of_attention_heads=8,
                 dim_feedforward=2048, num_of_encoder_leyrs=6, output_size=1, dropout_p=0.1, batch_first=True):
        
        # Inherit everything from the nn.Module
        super().__init__()
                
        # CNN as feature extraction 
        feature_per_res = int(num_features/4)
        out_features = [feature_per_res, feature_per_res, feature_per_res, feature_per_res]
        self.cnn = nn.Sequential(
                                nn.Conv1d(input_dim, 16, kernel_size=3, padding='same'),
                                nn.BatchNorm1d(num_features=16, affine=False),
                                nn.LeakyReLU(0.2,),
                                InceptionResBlock(16, kernels=conv_kernel_res, out_features=out_features, stride_size=1),
                                nn.LeakyReLU(0.2,))
        
        # Positional Encoding
        self.positional_encoder = PositionalEncoding(
                                    dim_model=num_features, dropout_p=dropout_p, max_len=1000)
        
        # Define Transfomer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model = num_features,
                                                        nhead = num_of_attention_heads,
                                                        dim_feedforward = dim_feedforward,
                                                        batch_first = batch_first)
        self.encoder_transformer = nn.TransformerEncoder(encoder_layer = self.encoder_layer,
                                                         num_layers = num_of_encoder_leyrs)
        
        # Define Linear 
        self.linear = nn.Sequential(
                        nn.Linear(num_features, output_size),
                        nn.Sigmoid())

    def forward(self, source):
        # T=S source and target sequence length, N batch size, E number of features 
        # source and target size is be (batch_size, sequence length)
        seq_len = source.shape[1]
        
        # permute before CNN, from [batch, seq_len, 1] to [batch, 1, seq_len]
        source = torch.permute(source, (0, 2, 1))
        # feature extraction
        source = self.cnn(source)
        # permute after cnn, from [batch, 1, seq_len] to [batch, seq_len, 1]
        source = torch.permute(source, (0, 2, 1))
        
        # positional encoding and get target mask
        source = self.positional_encoder(source) # [batch, seq, features]
        
        # transformer
        transformer_out = self.encoder_transformer(source) # [batch, seq, features]
        
        # linear and activation layer
        out = [self.linear(transformer_out[:,i,:]) for i in range(seq_len)]
        y_pred = torch.stack(out).T.permute(1,2,0)
        
        return y_pred
    
    
# Helper functions
class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by batch normalization and leakyrelu for unet.
    """

    def __init__(self, kernel, in_features, out_features, decoder=False):
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
    """
    def __init__(self, in_features, kernels=[15,17,19,21], out_features=[16,16,16,16], stride_size=1):
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
    
class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])
