# Arm_EMG_Data_Analysis
Testing features for arm EMG data classifications
### 1. Feature Definitions and Mathematical Expressions
In the table below are the names and mathematical definitions of the features I tried for the classification of arm surface EMG signals:
| Feature Name | Mathematical Expression|Definition |
| ------------ | -----------------------|-----------|
| Variance |$$\mathrm{Var} = \frac{1}{N-1} \sum(x_i-\bar x)^2$$ | Averaged squared deviation of the data from its mean (measures dispersion) |
| Waveform Length| $$WL=\sum \limits_{i=1}^{K-1}\mid {x}_{i+1}-{x}_i\mid$$|Sum of the absolute differences between adjacent signal point|
| Mean Absolute Value | $$\mathrm{MAV}=\frac{1}{\mathrm{K}}\sum \limits_{i=1}^{\mathrm{K}}\mid {x}_i\mid$$ | Average of the absolute values of the value points on the signal|
| Root Mean Square | $$\mathrm{RMS} = \sqrt{\frac{1}{K}\sum \limits_{i=1}^K{x}_i^2}$$ | The arithmetic mean of the squares of a set of values |
| Mean Frequency | $$\mathrm{MNF} = \frac{{\sum \limits_{j=1}^M} f_j P_j}{\sum \limits_{j=1}^M P_j}$$ | The ratio of the sum of the product of the electromyography signal power spectrum and the frequency to the sum of the spectral intensities|
| Frequency Ratio | $$\mathrm{FR}=\frac{\sum \limits_{j={LLC}}^{ULC} P_j}{\sum\limits_{j={LUC}}^{UUC} P_j}$$| The ratio of the low-frequency portion to the high-frequency portion of the myoelectric signal|

### 2. Histograms for Value Distribution of Single Features for Different Labels
![c1MAV](https://github.com/sinestrofeng2000/Arm_EMG_Data_Analysis/blob/main/Histogram_Features/histogram_plot_c1_MAV.png)
![c1MNF](https://github.com/sinestrofeng2000/Arm_EMG_Data_Analysis/blob/main/Histogram_Features/histogram_plot_c1_MNF.png)
![c1RMS](https://github.com/sinestrofeng2000/Arm_EMG_Data_Analysis/blob/main/Histogram_Features/histogram_plot_c1_RMS.png)
![c1FrequencyRatio](https://github.com/sinestrofeng2000/Arm_EMG_Data_Analysis/blob/main/Histogram_Features/histogram_plot_c1_frequencyRatio.png)
![c1Var](https://github.com/sinestrofeng2000/Arm_EMG_Data_Analysis/blob/main/Histogram_Features/histogram_plot_c1_var.png)
![c1Waveform](https://github.com/sinestrofeng2000/Arm_EMG_Data_Analysis/blob/main/Histogram_Features/histogram_plot_c1_waveformLength.png)

![c2MAV](https://github.com/sinestrofeng2000/Arm_EMG_Data_Analysis/blob/main/Histogram_Features/histogram_plot_c2_MAV.png)
![c2MNF](https://github.com/sinestrofeng2000/Arm_EMG_Data_Analysis/blob/main/Histogram_Features/histogram_plot_c2_MNF.png)
![c2RMS](https://github.com/sinestrofeng2000/Arm_EMG_Data_Analysis/blob/main/Histogram_Features/histogram_plot_c2_RMS.png)
![c2FrequencyRatio](https://github.com/sinestrofeng2000/Arm_EMG_Data_Analysis/blob/main/Histogram_Features/histogram_plot_c2_frequencyRatio.png)
![c2Var](https://github.com/sinestrofeng2000/Arm_EMG_Data_Analysis/blob/main/Histogram_Features/histogram_plot_c2_var.png)
![c2Waveform](https://github.com/sinestrofeng2000/Arm_EMG_Data_Analysis/blob/main/Histogram_Features/histogram_plot_c2_waveformLength.png)

![c3MAV](https://github.com/sinestrofeng2000/Arm_EMG_Data_Analysis/blob/main/Histogram_Features/histogram_plot_c3_MAV.png)
![c3MNF](https://github.com/sinestrofeng2000/Arm_EMG_Data_Analysis/blob/main/Histogram_Features/histogram_plot_c3_MNF.png)
![c3RMS](https://github.com/sinestrofeng2000/Arm_EMG_Data_Analysis/blob/main/Histogram_Features/histogram_plot_c3_RMS.png)
![c3FrequencyRatio](https://github.com/sinestrofeng2000/Arm_EMG_Data_Analysis/blob/main/Histogram_Features/histogram_plot_c3_frequencyRatio.png)
![c3Var](https://github.com/sinestrofeng2000/Arm_EMG_Data_Analysis/blob/main/Histogram_Features/histogram_plot_c3_var.png)
![c3Waveform](https://github.com/sinestrofeng2000/Arm_EMG_Data_Analysis/blob/main/Histogram_Features/histogram_plot_c3_waveformLength.png)

![c4MAV](https://github.com/sinestrofeng2000/Arm_EMG_Data_Analysis/blob/main/Histogram_Features/histogram_plot_c4_MAV.png)
![c4MNF](https://github.com/sinestrofeng2000/Arm_EMG_Data_Analysis/blob/main/Histogram_Features/histogram_plot_c4_MNF.png)
![c4RMS](https://github.com/sinestrofeng2000/Arm_EMG_Data_Analysis/blob/main/Histogram_Features/histogram_plot_c4_RMS.png)
![c4FrequencyRatio](https://github.com/sinestrofeng2000/Arm_EMG_Data_Analysis/blob/main/Histogram_Features/histogram_plot_c4_frequencyRatio.png)
![c4Var](https://github.com/sinestrofeng2000/Arm_EMG_Data_Analysis/blob/main/Histogram_Features/histogram_plot_c4_var.png)
![c4Waveform](https://github.com/sinestrofeng2000/Arm_EMG_Data_Analysis/blob/main/Histogram_Features/histogram_plot_c4_waveformLength.png)

### 3. Accuracies of KNN/LDA Models Trained on Single Features
![Accuracies](https://github.com/sinestrofeng2000/Arm_EMG_Data_Analysis/blob/main/AccuraciesPlot.png)

The best performing features for both the KNN and LDA models are waveform length and mean absolute value. Below are the confusion matrices for both Models trained on these 2 features.

![ConfMatKNN](https://github.com/sinestrofeng2000/Arm_EMG_Data_Analysis/blob/main/Confusion_knn_waveformLength_MAV.png)
![ConfMatLDA](https://github.com/sinestrofeng2000/Arm_EMG_Data_Analysis/blob/main/Confusion_lda_waveformLength_MAV.png)


