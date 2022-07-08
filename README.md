# R-peaks Detection ECG

The following Repo is a project in a course in the of Bio medical engineering faculty at the Technion, Machine Learning for Physiological Time Series Analysis (336018).
The project goal is to detect R-peaks in noisy ECG signal, and the frame for data generation was based on Laitala et all [1].
The input and output of the model is 1D tensor with length of 1000, continous values for input and target vector is boolean (1 for peaks, 0 for background).

After dataset creation, we compare multiple architectures:
 1. Bidirectional LSTM two-layers. 
 2. Bidirectional LSTM two-layers, with CNN as feature extraction stage.
 3. U-net with classical convulotion layers
 4. U-net with Res-Inception blocks
 5. Encoding tranformers, with CNN as feature extraction stage.
 6. Encoding tranformers, with Res-Inception blocks as feature extraction stage.


## Data Sources
The input is MIT-BIH database: 
1. MIT-BIH Arrhythmia database, includes "clean" ECG signal [2]
2. MIT-BIH Noise Stress database, includes only ECG "noise" [3]
For test set:
3. 







## Reference

[1] Juho Laitala, Mingzhe Jiang, Elise Syrjälä, Emad Kasaeyan Naeini, Antti Airola, Amir M. Rahmani, Nikil D. Dutt, and Pasi Liljeberg. 2020. Robust ECG R-peak detection using LSTM. Proceedings of the 35th Annual ACM Symposium on Applied Computing. Association for Computing Machinery, New York, NY, USA, 1104–1111. https://doi.org/10.1145/3341105.3373945

[2] https://physionet.org/content/mitdb/1.0.0/

[3] https://physionet.org/content/nstdb/1.0.0/

