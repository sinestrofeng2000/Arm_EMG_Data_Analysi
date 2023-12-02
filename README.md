# Arm_EMG_Data_Analysis
Testing features for arm EMG data classifications
### Feature Definitions and Mathematical Expressions
In the table below are the names and mathematical definitions of the features I tried for the classification of arm surface EMG signals:
| Feature Name | Mathematical Expression|Definition |
| ------------ | -----------------------|-----------|
| Variance |$$\mathrm{Var} = \frac{1}{N-1} \sum(x_i-\bar x)^2$$ | Averaged squared deviation of the data from its mean (measures dispersion) |
| Waveform Length| $$WL=\sum \limits_{i=1}^{K-1}\mid {x}_{i+1}-{x}_i\mid$$|Sum of the absolute differences between adjacent signal point|
| Mean Absolute Value | $$\mathrm{MAV}=\frac{1}{\mathrm{K}}\sum \limits_{i=1}^{\mathrm{K}}\mid {x}_i\mid$$ | Average of the absolute values of the value points on the signal|
| Root Mean Square | $$\mathrm{RMS} = \sqrt{\frac{1}{K}\sum \limits_{i=1}^K{x}_i^2}$$ | The arithmetic mean of the squares of a set of values |
| Mean Frequency | $$\mathrm{MNF} = \frac{{\sum \limits_{j=1}^M} f_j P_j}{\sum \limits_{j=1}^M P_j}$$ | The ratio of the sum of the product of the electromyography signal power spectrum and the frequency to the sum of the spectral intensities|
| Frequency Ratio | $$\mathrm{FR}=\frac{\sum \limits_{j={LLC}}^{ULC} P_j}{\sum\limits_{j={LUC}}^{UUC} P_j}$$| The ratio of the low-frequency portion to the high-frequency portion of the myoelectric signal|

### Histograms for Value Distribution of Single Features for Different Labels

