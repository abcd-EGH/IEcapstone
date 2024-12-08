# Industrial Engineering Capstone at Hanyang University
24-2학기 산업공학캡스톤PBL 수업의 일환으로, Ensemble of Sparsely connected RNN and Autoencoder for Anomaly Detection in Financial Time Series을 주제로 실험을 진행하였습니다.

PyTorch framework로 base & advanced model을 구현한 코드는 https://github.com/abcd-EGH/srnn-ae 에서 확인 가능합니다.

Base model 구현을 위해 다음 논문과 GitHub를 참고하였습니다: **Tung Kieu, Bin Yang, Chenjuan Guo, Christian S. Jensen, Outlier Detection for Time Series with Recurrent Autoencoder Ensembles, IJCAI 2019.** https://doi.org/10.24963/ijcai.2019/378<br> https://github.com/tungk/OED

This project is part of the 24-2 Semester **Industrial Engineering Capstone PBL** course. The experiment focuses on the topic: **Ensemble of Sparsely Connected RNN and Autoencoder for Anomaly Detection in Financial Time Series**.

The code for both the base and advanced models implemented using the PyTorch framework is available at https://github.com/abcd-EGH/srnn-ae.

To implement the base model, the following paper and GitHub repository were referenced:  
**Tung Kieu, Bin Yang, Chenjuan Guo, Christian S. Jensen, Outlier Detection for Time Series with Recurrent Autoencoder Ensembles, IJCAI 2019.** https://doi.org/10.24963/ijcai.2019/378<br>
https://github.com/tungk/OED

## Abstact

### Dataset
This implementation uses the following publicly available dataset:
- NAB (Numenta Anomaly Benchmark): For anomaly detection in time series data.

To use the dataset, please refer to the original paper or download them directly from the provided sources.

E.g. Download w/ Git Commend
```bash
git clone https://github.com/numenta/NAB.git
```
- https://github.com/numenta/NAB
- https://doi.org/10.5281/zenodo.1040335

### Used Data & Seasonal Trend Decomposition
- realKnownCause/**machine_temperature_system_failure**.csv
    <img src="EDA\machine_temperature_seasonal_decomposition.png" width="800" height="400"/>
- realAWSCloudwatch/**ec2_cpu_utilization_825cc2**.csv
    <img src="EDA\ec2_cpu_utilization_seasonal_decomposition.png" width="800" height="400"/>
- artificialWithAnomaly/**art_daily_jumpsup**.csv
    <img src="EDA\art_daily_jumps_up_seasonal_decomposition.png" width="800" height="400"/>

### Defaults for Hyper-Parameter
 - **N = 10**, Number of AutoEncoders in Ensemble
 - **input_size = 1**, Input size (e.g., single time series)
 - **hidden_size = 8**, Size of the hidden layer in the RNN
 - **output_size = 1**, Output size
 - **num_layers = 1**, Number of RNN layers (not explicitly mentioned in the paper)
 - **limit_skip_steps = 10**, Maximum number of skip connections L (randomly chosen between 1 and 10)
 - **learning_rate = 1e-3**, Learning rate for the optimizer
 - **l1_lambda = 1e-5**, Regularization parameter for L1 penalty (As an exception, 1e-3 for Residual)
 - **window_size = 36**, Window size for time series, One day for every 288 (not explicitly mentioned in the paper)
 - **num_epochs = 1000**, Number of training epochs (not explicitly mentioned in the paper)
 - **random_seed = 777**, Random seed for reproducibility 
 - stride = **1**

## Models used in each hypothesis
**Details of following models can be found at https://github.com/abcd-EGH/srnn-ae.**
### H1
- **ESLAE** (**E**nsemble of **S**parsely connection **L**STM and **A**uto**E**ncoder)
- Base Model, Sparsely Connections

### H2
- **ERSLAE** (**E**nsemble of **Residual** & **S**parsely connection **L**STM and **A**uto**E**ncoder)
- Advanced Model, Residual & Sparsely Connections

### H3
- **ECSLAE** (**E**nsemble of **C**oncatenation-based skip (Encoder-Decoder) & **S**parsely connection **L**STM and **A**uto**E**ncoder)
- Advanced Model, Concatenation-based skip & Sparsely Connections

### H4
- **EVSLAE** (**E**nsemble of **V**ariable-skip (Encoder-Decoder) & **S**parsely connection **L**STM and **A**uto**E**ncoder)
- Advanced Model, Variable-skip & Sparsely Connections

### H5
- **ESBLAE** (**E**nsemble of **S**parsely connection **B**i-directional **L**STM and **A**uto**E**ncoder)
- Advanced Model, Bi-directional LSTM

### H6
- **ECSLAE** (**E**nsemble of **A**ttention & **S**parsely connection **L**STM and **A**uto**E**ncoder)
- Advanced Model, Attention

## Results

### Quantitative Results
#### Table 1: Performance on **machine_temperature_system_failure**
| Model          | Precision | Recall  | F1-Score |
|----------------|-----------|---------|----------|
| Base           | 0.7612    | 0.3810  | 0.5078   |
| Residual       | 0.7683    | 0.3845  | 0.5125   |
| Concatenation  | 0.7110    | 0.3558  | 0.4743   |
| VariableSkip   | 0.8035    | 0.4021  | 0.5360   |
| Bi-directional | 0.7410    | 0.3708  | 0.4943   |
| Attention      | 0.5040    | 0.2522  | 0.3362   |

#### Table 2: Performance on **ec2_cpu_utilization_825cc2**
| Model          | Precision | Recall  | F1-Score |
|----------------|-----------|---------|----------|
| Base           | 0.5347    | 0.3149  | 0.3963   |
| Residual       | 0.5446    | 0.3207  | 0.4037   |
| Concatenation  | 0.5495    | 0.3236  | 0.4073   |
| VariableSkip   | 0.5396    | 0.3178  | 0.4000   |
| Bi-directional | 0.5347    | 0.3149  | 0.3963   |
| Attention      | 0.6238    | 0.3673  | 0.4624   |

#### Table 3: Performance on **art_daily_jumpsup**
| Model          | Precision | Recall  | F1-Score |
|----------------|-----------|---------|----------|
| Base           | 0.5743    | 0.2878  | 0.3835   |
| Residual       | 0.5545    | 0.2779  | 0.3702   |
| Concatenation  | 0.5842    | 0.2928  | 0.3901   |
| VariableSkip   | 0.5594    | 0.2804  | 0.3736   |
| Bi-directional | 0.5842    | 0.2928  | 0.3901   |
| Attention      | 0.5050    | 0.2531  | 0.3372   |

### Qualitative Results
_In each hypothesis ipynb file, you can find qualitative results such as **Reconstructed data and errors**, **AUC-ROC curves**, and so on._

## Conclusion

### Residual Connection
- **Observation**: Output data closely follows the shape of input data, achieving high anomaly detection performance (hereafter referred to as "performance") even for time series with high irregularity.
- **Interpretation**: Residual Connection alleviated the vanishing gradient problem while enhancing generalization performance.

### Concatenation
- **Observation**: Exhibited lower performance for time series with high irregularity but higher performance for time series with consistent amplitude.
- **Interpretation**: The decoder was able to learn not only compressed information from the encoder but also specific time step information, enabling it to capture large and periodic variations effectively.

### Variable Skip
- **Observation**: Achieved strong performance across both highly irregular time series and time series with large, consistent amplitudes.
- **Interpretation**: Ensuring that the skip length \( L \) was evenly distributed across all autoencoders (from 1 to \( n \)) improved generalization performance.

### Bi-directional and Attention Mechanisms
- **Observation**: Did not achieve consistently high performance overall.
  - **Bi-directional**: Bidirectional learning tended to capture unnecessary information in irregular time series, leading to vanishing gradient issues.
  - **Attention**: For irregular time series, unnecessary attention weights were assigned to noise, while in simpler time series, it struggled to identify "important parts" to learn from.

---

### Variable Skip (Detailed Analysis)
(**L**: Length of skip connection)
#### Expected Results
- **Short \( L \)**: Insufficient learning of time series information.
- **Long \( L \)**: Better learning of long-term information and periodic patterns.

#### Experimental Results
- **Observation**: Shorter \( L \) resulted in output data closely following the shape of input data, with better performance overall.

#### Problem Analysis
- Constructing the ensemble model with a single class led to the following issues:
  1. **Interdependence during Gradient Descent**: The weights of each model influenced the learning process of other models.
  2. **Interference during Backpropagation**: Overlapping gradients caused interference in the learning paths of individual models.

#### Solution
- Train models with different \( L \) values independently and use only the averaged outputs for predictions. This approach avoids interdependence and interference during the learning process.


## REFERENCE
1. Kieu, T., Yang, B., Guo, C., & Jensen, C. S. (2019). Outlier detection for time series with recurrent autoencoder ensembles. Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence (IJCAI), 2725–2732. https://doi.org/10.24963/ijcai.2019/378
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770–778. https://doi.org/10.1109/CVPR.2016.90
3. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. Proceedings of the International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), 234–241. https://doi.org/10.1007/978-3-319-24574-4_28
4. Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. IEEE Transactions on Signal Processing, 45(11), 2673–2681. https://doi.org/10.1109/78.650093
5. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems (NeurIPS), 30, 5998–6008. https://arxiv.org/abs/1706.03762