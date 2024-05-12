# DENFIS: Dynamic Evolving Neural-Fuzzy Inference System
This project implements the Dynamic Evolving Neural-Fuzzy Inference System (DENFIS) for predicting time-series data. DENFIS is an advanced approach combining neural networks and fuzzy logic to provide accurate and efficient predictions.

## DENFIS
DENFIS, or Dynamic Evolving Neural-Fuzzy Inference System, is a method that combines elements of neural networks and fuzzy logic to predict outcomes in time-series data. It works by taking historical data that changes over time and learning patterns from it. Using this knowledge, it makes predictions about future data points. DENFIS is especially useful because it can adapt to new data as it becomes available, making it a dynamic and evolving system. This adaptability makes it well-suited for tasks where conditions change frequently, such as financial market predictions or weather forecasting.

The DENFIS model first introduced by [Kasabov, 2002](https://doi.org/10.1109/91.995117) in 2002. Take a look at Figure below, showing the architechuter of DENFIS.

![Figure 2](https://i.ibb.co/MNGrWrP/Screenshot-from-2023-11-20-22-13-33.png)

### DENFIS Layers
This is the explanation for each layer:

- **Layer 1:** Membership function layer. Turning the input according to the membership function formula. For DENFIS, the membership function is generated when we train the model with the training dataset. The detail will be explained in the next section.

- **Layer 2:** The output of each node of this layer is the product of all the incoming signals.
  $$
  O_{2,i} = w_i = \mu_{A_i}(x) \cdot \mu_{B_i}(y), \quad i = 1, 2
  $$

- **Layer 3:** The outputs of this layer are the normalization of the incoming signals.
  $$
  O_{3,i} = \frac{w_i}{w_1 + w_2}, \quad i = 1, 2
  $$

- **Layer 4:** This is the layer where least square is used to get the parameter for the weight.
  $$
  O_{4,1} = \tilde{w}_i f_i = \tilde{w}_i (p_x x + q_i y + r_i)
  $$

- **Layer 5:** The output of this layer is computed as the summation of all the incoming signals.
  $$
  \sum_i \tilde{w}_i f_i
  $$

### Evolving Clustering Method (ECM)

The Evolving Clustering Method (ECM) is used to subdivide the input set and determine the position of each data in the input set. This is the part when membership function is generated when the model is at the training phase.

Here are the simplified steps for generating said membership function:

1. Set a threshold for a cluster diameter threshold.
2. Iterate for all the training data that are given.
3. Put new data to a cluster, if the distance between is less or equal to the diameter threshold. Otherwise, create a new cluster.
4. For each of the clusters generated, a triangular membership function is defined by this formula:
   $$
   \mu(\hat{x}) = mf(\hat{x}, a, b, c) = \max\left(\min\left(\frac{(\hat{x}-a)}{(b-a)}, \frac{(c-\hat{x})}{(c-b)}\right), 0\right),
   $$
   where:
   - $\hat{x}$ will be the testing input
   - $a = b - (d \times \text{diameter\_threshold})$
   - $c = b + (d \times \text{diameter\_threshold})$
   - $d$ is a parameter of the width of the triangular function (1.2-2.0)

## Dataset

Trust stock used for the project:
- **Trust Stock**: BlackRock Multi-Sector Income Trust (BIT)
- **Period**: Using the full dataset from 2013-02-26 until the current date.
- **Trust Stock Description**: BlackRock Multi-Sector Income Trust's (BIT) (the 'Trust') primary investment objective is to seek high current income [BlackRock Company, 2023](https://www.blackrock.com/us/individual/products/249839/blackrock-multisector-income-trust-fund).

## Benchmark Method
The RMSE error will be evaluated for determining which model version of model perform the best.

For splitting the dataset, this project use a blocking time series split. Blocking time series split will seperate the dataset into five part of cross folding.

First four of those cross folding will be used to determined the parameter of the model. While the last fold will be used to be tested and to be evaluated.

After the prediction is retrieved, it will be put into a simulation to determined the portofolio returns. The best model should able to return high portofolio returns.

## Implementation
There is little to none DENFIS implementation or library that are readily available and/or easy to use for this project.

Therefore, in this project DENFIS is going to be implemented **from scratch**, only using basic library like numpy.


# References
- DENFIS:  N. K. Kasabov and Qun Song, "DENFIS: dynamic evolving neural-fuzzy inference system and its application for time-series prediction," in IEEE Transactions on Fuzzy Systems, vol. 10, no. 2, pp. 144-154, April 2002, doi: 10.1109/91.995117.
- Stock Trading: K. K. Ang and C. Quek, "Stock Trading Using RSPOP: A Novel Rough Set-Based Neuro-Fuzzy Approach," in IEEE Transactions on Neural Networks, vol. 17, no. 5, pp. 1301-1315, Sept. 2006, doi: 10.1109/TNN.2006.875996.