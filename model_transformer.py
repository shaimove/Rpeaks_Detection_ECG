# model_transformer.py
import torch
import torch.nn as nn
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Simple Encoding Transformer model, without preprocessing with linear layer as post processing
class transformer_model(nn.Module):
    """
    Simple Encoding Transfomer with Positional Encoding.
    No preprocessing, with linear layer as post processing.
    """
    
    def __init__(self, 
                 input_dim=1, 
                 cnn_kernel=3, 
                 num_features=128, 
                 num_of_attention_heads=8,
                 dim_feedforward=2048, 
                 num_of_encoder_leyrs=6, 
                 output_dim=1, 
                 dropout_p=0.1, 
                 batch_first=True):
        """
        Model definition

        Parameters
        ----------
        input_dim : number of features per ECG signal, in input
            int. The default is 1.
        cnn_kernel : kernel size of CNN for feature extraction
            int. The default is 3.
        num_features : number of features for signal in the sequeence
            int. The default is 128.
        num_of_attention_heads : number of attention heads in the transformer
            int. The default is 8.
        dim_feedforward : feedforward features in the transformer
            int. The default is 2048.
        num_of_encoder_leyrs : number of encoding transformer layers
            int. The default is 6.
        output_dim : number of features per ECG signal, in output
            int. The default is 1.
        dropout_p : dropout probability in positional encoding
            float < 1. The default is 0.1.
        batch_first : is batch the first axis
            boolean. The default is True.

        """
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
                        nn.Linear(num_features, output_dim),
                        nn.Sigmoid())

    def forward(self, source):
        """
        Forward function

        Parameters
        ----------
        source : Tensor of ECG signal
            size [batch_size, sequence, signal_length, input_dim].

        Returns
        -------
        y_pred : Tensor of peaks in ECG signal, range [0,1]
            size [batch_size, sequence, signal_length, output_dim].

        """
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

# Encoding Transformer with Res-Inception
class transformer_inception_model(nn.Module):
    """
    Encoding Transfomer with Positional Encoding, with Res-Inception block
    Res-Inception block as preprocessing, with linear layer as post processing.
    """
    
    def __init__(self, 
                 input_dim=1, 
                 conv_kernel_res=[15,17,19,21], 
                 num_features=128, 
                 num_of_attention_heads=8,
                 dim_feedforward=2048, 
                 num_of_encoder_leyrs=6, 
                 output_dim=1, 
                 dropout_p=0.1, 
                 batch_first=True):
        """
        Model definition

        Parameters
        ----------
        input_dim : number of features per ECG signal, in input
            int. The default is 1.
        conv_kernel_res : list of convolution kernels applied in every block,
            different resolutions applied in every block.
            list of int. The default is [15,17,19,21].
        num_features : number of features for signal in the sequeence
            DESCRIPTION. The default is 128.
        num_of_attention_heads : number of attention heads in the transformer
            int. The default is 8.
        dim_feedforward : feedforward features in the transformer
            int. The default is 2048.
        num_of_encoder_leyrs : number of encoding transformer layers
            int. The default is 6.
        output_dim : number of features per ECG signal, in output
            int. The default is 1.
        dropout_p : dropout probability in positional encoding
            float < 1. The default is 0.1.
        batch_first : is batch the first axis
            boolean. The default is True.

        """
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
                        nn.Linear(num_features, output_dim),
                        nn.Sigmoid())

    def forward(self, source):
        """
        Forward function

        Parameters
        ----------
        source : Tensor of ECG signal
            size [batch_size, sequence, signal_length, input_dim].

        Returns
        -------
        y_pred : Tensor of peaks in ECG signal, range [0,1]
            size [batch_size, sequence, signal_length, output_dim].

        """
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
    
    
#%% Helper functions
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
    

class PositionalEncoding(nn.Module):
    def __init__(self, 
                 dim_model, 
                 dropout_p, 
                 max_len):
        """
        

        Parameters
        ----------
        dim_model : TYPE
            DESCRIPTION.
        dropout_p : TYPE
            DESCRIPTION.
        max_len : TYPE
            DESCRIPTION.

        """
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
        """
        

        Parameters
        ----------
        token_embedding : torch.tensor
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])
