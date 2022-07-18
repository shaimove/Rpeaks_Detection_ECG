# model_lstm.py
import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Simple LSTM - no preprocessing, with linear layer as post process
class lstm_model(nn.Module):
    """
    LSTM model, seq2seq architecture to predict R-peaks in ECG signal.
    No preprocessing, with linear layer as post processing
    """
    def __init__(self,
                 input_dim=1, 
                 hidden_size=64, 
                 output_dim=1, 
                 n_layers=2, 
                 dropout=0.2,
                 bidirectional=True):
        """
        Model definition

        Parameters
        ----------
        input_dim : number of features per ECG signal, in input
            int. The default is 1.
        hidden_size : size of hidden state of LSTM
            int. The default is 64.
        output_dim : number of features per ECG signal, in output.
            int. The default is 1.
        n_layers : number of LSTM layers
            int. The default is 2.
        dropout : dropout probability in LSTM.
            float < 1. The default is 0.2.
        bidirectional : use bidirectional LSTM or unidirectional LSTM.
            boolean. The default is True.

        """
        # Inherit everything from the nn.Module
        super().__init__()
        
        n_directions = 2 if bidirectional else 1
            
        # Define LSTM
        self.lstm = nn.LSTM(input_size = input_dim, 
                            hidden_size = hidden_size, 
                            num_layers = n_layers, 
                            dropout = (0 if n_layers == 1 else dropout), 
                            batch_first=True,
                            bidirectional = bidirectional) 
        
        # Define Linear 
        self.linear = nn.Sequential(
                                nn.Linear(hidden_size*n_directions, output_dim),
                                nn.Sigmoid())
        
    def forward(self, input_seq, hidden=None):
        """
        Forward function
        
        Parameters
        ----------
        input_seq : Tensor of ECG signal
            size [batch_size, sequence, signal_length, input_dim].
        hidden : Tensor of short-long term memory
            Tensor [n_layers*n_directions, batch, hidden]. The default is None.

        Returns
        -------
        y_pred : Tensor of peaks in ECG signal, range [0,1]
            size [batch_size, sequence, signal_length, output_dim].

        """
        # Extract batch_size, Input: [batch, seq, hidden]
        self.seq_length = input_seq.size(1)
        
        # fisrt direction LSTM, hidden ([2*2, batch, hidden], [2*2, batch, hidden]), lstm_out [batch, seq, 2*hidden]
        lstm_out, hidden = self.lstm(input_seq, hidden)

        # linear and activation layer
        out = [self.linear(lstm_out[:,i,:]) for i in range(self.seq_length)]
        y_pred = torch.stack(out).T.permute(1,2,0)
        
        return y_pred
    
# LSTM model with CNN as preprocessing, and linear layer as postprocessing
class cnn_lstm_model(nn.Module):
    """
    LSTM model, seq2seq architecture to predict R-peaks in ECG signal.
    CNN as preprocessing, with linear layer as post processing
    """
    def __init__(self,
                 conv_input_dim = 1,
                 conv_kernel = [3,3,3,3,3],
                 conv_feature = [16,32,32,16,1],
                 hidden_size_lstm = 64, 
                 n_layers_lstm = 2, 
                 dropout_lstm = 0.2,
                 bidirectional = True,
                 output_dim = 1):
        """
        Model definition

        Parameters
        ----------
        conv_input_dim : number of features per ECG signal, in input
            int. The default is 1.
        conv_kernel : list of convolution kernels
            list of int. The default is [3,3,3,3,3].
        conv_feature : list of convolution features, len(conv_feature) = len(conv_kernel)
            list of int. The default is [16,32,32,16,1].
        hidden_size_lstm : size of hidden state of LSTM
            int. The default is 64.
        n_layers_lstm : number of LSTM layers
            int. The default is 2.
        dropout_lstm : dropout probability in LSTM.
            float < 1. The default is 0.2.
        bidirectional : use bidirectional LSTM or unidirectional LSTM.
            boolean. The default is True.
        output_dim : number of features per ECG signal, in output.
            int. The default is 1.

        """
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
                          dropout = (0 if n_layers_lstm == 1 else dropout_lstm), batch_first=True,
                          bidirectional = bidirectional) 
        
        # Define Linear 
        self.linear = nn.Sequential(
                        nn.Linear(hidden_size_lstm*n_directions, output_dim),
                        nn.Sigmoid())

        
    def forward(self, input_seq, hidden=None):
        """
        Forward function
        
        Parameters
        ----------
        input_seq : Tensor of ECG signal
            size [batch_size, sequence, signal_length, conv_input_dim].
        hidden : Tensor of short-long term memory
            size [n_layers*n_directions, batch, hidden]. The default is None.

        Returns
        -------
        y_pred : Tensor of peaks in ECG signal, range [0,1]
            size [batch_size, sequence, signal_length, output_dim].

        """
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

