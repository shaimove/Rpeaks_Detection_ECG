# R-peaks Detection ECG

The following Repo is part of a project in a course in the Biomedical engineering faculty at the Technion, Machine Learning for Physiological Time Series Analysis (336018).

ECG is the measurement of your heart's electrical activity, enabling physicians to detect many diseases related to the heart's electrical function. Several electrodes are positioned on the patient's chest in different locations, and we measure the voltage difference between different electrodes. 
Usually, when we perform the test in a hospital or other medical facility, we do it with multiple sensors, position the electrode accurately on the body. At the end of measurements in a medical facility, we get a very "clean" signal, almost free of noise, and in the past; Many classical algorithms for R peaks detection were developed and worked well for clean ECG signals. 

The problem is that we want to perform the test without scheduling an appointment in a medical facility, and to measure the ECG signal easily at home, even on a wearable device. It will reduce costs and enable us to detect cardiac problems long before it becomes severe at home at a low cost.
The problem with wearable devices is the level of noise. There are many noise sources such as motions of the patient, changes in conduction, dry electrodes, electrical noise from the battery or line, etc. 
We need a robust algorithm to detect R-peaks on edge devices to overcome high levels and sources of noise. 

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

For test set, we used four other database and they are described in the Test Models section.

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

The new loss function can be written as follows:

![loss function](https://user-images.githubusercontent.com/47494709/183284535-8da2ad58-06d2-4c0d-9213-48f6f2e9a504.png)

## Peak Labeling
In ECG signal, we have multiple peaks, most of which are "Normal" peaks. But there are also many abnormal peaks; for different Arthymes, we have peaks that look like normal peaks (in the sense of a peak in a 1D signal). In the data generation process, we exlude signals with abnormal peaks; they might be an indication of a disease or not, nevertheless, they exist in the ECG signal. 

After evaluating all models on the validation set, we changed the labeling of a few abnormal peaks to normal peaks, so the algorithm could train to detect them, and we got improved results. We changed the following peak types to "normal" peak. 

![image](https://user-images.githubusercontent.com/47494709/178142739-8b98f5d8-9b71-45d7-bf68-a5cc8117d7c9.png)


## Model output example
The following examples shows the signals; in Red the original noisy ECG signal, in Blue the taget signal (GT), in purple the output probability. 

![image](https://user-images.githubusercontent.com/47494709/177945079-e2679d1d-6b59-4ad5-ae1b-d2c1001c3f16.png)

## Results on Validation set
Before creating the data loaders, we splited the patients into two differnet sets, one for training and one for validation. The following results were achieved:

![validation](https://user-images.githubusercontent.com/47494709/183287686-27d9c634-7883-42a7-816d-e631f135ba4c.png)

For the validation set, we can see that the Transformers architectures achieved the best results in terms of Recall and Specificity compared to LSTM and U-net architectures. Res-Inception blocks improved the performance for all architectures, and changing the labeling of the peaks enhanced the performance. In addition, all deep-learning methods are superior to the classical algorithm (G-QRS algorithm is implemented in the wfdb library). 

## Results on Test databases
In the testing set, we could use the MTI-BIH database and use information from a separate group of patients, but in order to test if our trained models can be generalized to "real-world" data, we choose to use external databases. The database contains a variety of different types of ECG signals: normal sinus rhythm (nsrdb), arrhythmia (svdb & incartdb), ST and T morphology changes (edb), and long-term measurements (nsrdb).

The following databases were used for testing the trained models, no artifical noise was added to signals:
1. MIT-BIH Normal Sinus Rhythm Database [4], will be marked as "nsrdb"
2. MIT-BIH Supraventricular Arrhythmia Database [5], will be marked as "svdb"
3. St Petersburg INCART 12-lead Arrhythmia Database [6], will be marked as "incartdb"
4. European ST-T Database [7], will be marked as "edb"

The following results were achieved in percentage for different models and different databases:

Recall:

![Recall Test](https://user-images.githubusercontent.com/47494709/183290116-3d66d109-26f3-417f-a797-098615a69ba5.png)

Specificity:

![Specificity test](https://user-images.githubusercontent.com/47494709/183290120-24757cd1-60f5-4c83-b2dd-037d84a23f8f.png)

Next, we wanted to see if adding abnormal peaks could improve the results, only for the last architecture: Encoding transformers, with Res-Inception block as feature extraction stage. We labeled "Normal" peaks as peaks in all test databases, disregarding all other peaks. The table below shows that the inclusion of abnormal peaks didn't improve the results. 

![label correction](https://user-images.githubusercontent.com/47494709/183293856-a2bfa8c4-4b8f-4199-8841-0ba4a6fd7381.png)

## Discussion

The project's goal was mainly to test whether transformers architectures, a more recent architecture that showed improved results in computer vision and NLP, can also benefit R-peaks in 1D ECG signal. During training and validation, we saw that transformers architectures achieved higher results compared to LSTMs and U-net architectures. 

The results decreased significantly when we evaluated the models on other databases without adding noise. This decrease can be explained by the "unnatural" nature of the noise we added. Muscle Artifacts and Base Wander amplitudes can be correlated to a "clean" ECG signal, so the noise-adding process wasn't correct. The models were trained with high levels of noise, which may not represent the noise level of a real ECG signal. Another reason could be that MIT-BIH doesn't contain a variety of ECG signals, for example, different arrhythmias and quality of ECG signal. 

Nevertheless, in both validation and test databases, all deep-learning approaches outperform the classical method that was used (GQRS algorithm). Future work should be done to improve R-peaks detection using deep-learning models.

Sevral improvements can be done:
1. Use "real" noisy ECG signal instead synthetic noise. 
2. Use multiple databases in the training and validation process. 
3. Improve peaks labeling methods
4. Use longer length for detection (we used fs=250Hz, 1000 length vector, only 4 seconds)

## Reference

[1] Juho Laitala, Mingzhe Jiang, Elise Syrjälä, Emad Kasaeyan Naeini, Antti Airola, Amir M. Rahmani, Nikil D. Dutt, and Pasi Liljeberg. 2020. Robust ECG R-peak detection using LSTM. Proceedings of the 35th Annual ACM Symposium on Applied Computing. Association for Computing Machinery, New York, NY, USA, 1104–1111. https://doi.org/10.1145/3341105.3373945

[2] https://physionet.org/content/mitdb/1.0.0/

[3] https://physionet.org/content/nstdb/1.0.0/

[4] https://www.physionet.org/content/nsrdb/1.0.0/

[5] https://physionet.org/content/svdb/1.0.0/

[6] https://physionet.org/content/incartdb/1.0.0/

[7] https://physionet.org/content/edb/1.0.0/

