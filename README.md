# R-peaks Detection ECG

The following Repo is part of a project in a course in the Biomedical engineering faculty at the Technion, Machine Learning for Physiological Time Series Analysis (336018).

The project's goal is to detect R-peaks in noisy ECG signals; The frame for data generation is based on Laitala et al. l [1]. In this project, we compared different architectures of RNN and compared them in terms of Recall and Specificity. In this project, we also introduced a new loss function and changed the R-peak labeling convention to improve results. 

The model's input and output is a 1D tensor with a length of 1000, and continuous values for input (noisy ECG). The target vector is a probability tensor, threshold to boolean tensor (1 for peaks, 0 for background).

After dataset creation, we compare the following architectures:
 1. Bidirectional LSTM two-layers. 
 2. Bidirectional LSTM two-layers, with CNN as feature extraction stage.
 3. U-net with classical convolution layers
 4. U-net with Res-Inception blocks
 5. Encoding transformers, with CNN as feature extraction stage.
 6. Encoding transformers, with Res-Inception block as feature extraction stage.
 7. Encoding transformers, with Res-Inception block as feature extraction stage, with modified labeling convention.


## Data Sources and Data creation
The input is MIT-BIH database: 
1. MIT-BIH Arrhythmia database, includes "clean" ECG signal [2]
2. MIT-BIH Noise Stress database, includes only ECG "noise" [3]

For test set, we used new data source:
3. 

The data creation method used for this project was the one used in Laitala et al. [1], and we used part of their function for implementation, and can be found in:
https://github.com/jtlait/ecg2rr.

The MIT-BIH Arrhythmia database contains 48 recordings from 47 patients; the recording was split into training and validation groups. In every data generation, we randomly sampled 1000 length sequences from the recording, and if all beats were "Normal", we normalized the signals to [-1,1]. Then we selected if to add Base Wander noise, Muscle Artifact noise, or both from the MIT-BIH Noise Stress database. We sampled the signals with random amplitude and added to the noise a synthetic electrical noise with random amplitude, with sinusoidal noise of 60Hz. Then we combine the signal and noise and renormalize it to [-1,1] range.

## Peak Expanding
In peak detection, the input and output signal has a length of 1000, but only 4-5 peaks, which means we have 0.4-0.5% of positive cases - unbalanced data.
We need to choose the appropriate loss function, we can use Binary Cross-Entropy (BCE), but this loss function will not deliver information about the distance between the predicted and true peak position. 

We can use a function that measures this distance, but the calculation becomes problematic in cases of multiple outputs far from the true peak. Hence, it is a common practice to use "peak expanding" - meaning change the values of the target tensor around to boolean peak, to "1". for example, define 5 points around the peak as "1", so we can measure the proximity of the detected peak compare to the real peak. Using Recall and Specificity, we can estimate the overlap between detected and true peaks:

![image](https://user-images.githubusercontent.com/47494709/178142049-2329a558-287d-4595-8ab0-da3ded5fc859.png)


## Loss Function
Even after peak expanding, the database is highly unbalanced (2-3% of target tensor is positive), so we need to adapt the loss function from regular BCE. When I tried to train the basic LSTM model with BCE, the model didn't converge; 

The new loss function comprises weighted BCE with weights according to the data imbalance. Since most of the tensor should be "zeros", I added a loss for all probabilities so that the expected output would be closer to zeros. This loss ensures that the average output is lower and increases the "certainty" of classification without reducing the Recall. Last, I added L1 regularization for the model's parameters.


## Peak Labeling



### Results
The following examples show the signals; in Red the original noisy ECG signal, in Blue the taget signal (GT), in purple the output probability. 

![image](https://user-images.githubusercontent.com/47494709/177945079-e2679d1d-6b59-4ad5-ae1b-d2c1001c3f16.png)


## Reference

[1] Juho Laitala, Mingzhe Jiang, Elise Syrjälä, Emad Kasaeyan Naeini, Antti Airola, Amir M. Rahmani, Nikil D. Dutt, and Pasi Liljeberg. 2020. Robust ECG R-peak detection using LSTM. Proceedings of the 35th Annual ACM Symposium on Applied Computing. Association for Computing Machinery, New York, NY, USA, 1104–1111. https://doi.org/10.1145/3341105.3373945

[2] https://physionet.org/content/mitdb/1.0.0/

[3] https://physionet.org/content/nstdb/1.0.0/

