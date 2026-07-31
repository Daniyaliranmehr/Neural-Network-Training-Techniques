<h1 align="center">Explore Neural Network Training Techniques</h1>

# Project Overview

# Project Structure

# File Descriptions

# Data Analysis and Preparation

## 1. Data Understanding

Initial exploration of the dataset to understand its structure, feature types, data quality aspects, and basic statistical properties. This step establishes a foundational understanding of the data before conducting further analysis, preprocessing, and model development.

### 1.1. Dataset Structure

The dataset used in this project is the **Statlog (Shuttle) dataset**, a multiclass classification dataset derived from NASA space shuttle simulations. It contains numerical measurements describing different shuttle conditions, with the goal of classifying each observation into one of several predefined classes.

🔗 [Statlog (Shuttle) Dataset - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/148/statlog)

#### Dataset Information

| Property | Description |
|:---:|:---:|
| Number of instances | 43,500 |
| Number of features | 9 |
| Target variable | Feature 9 |
| Task type | Multiclass Classification |
| Data type | Numerical |

#### Features

The dataset contains nine numerical features representing shuttle sensor measurements. The original dataset does not provide descriptive feature names, so features are referenced by their column indices.

#### Target Variable

| Target | Description |
|:---:|:---:|
| target | Class label representing the shuttle condition |

The target variable contains multiple classes ranging from 1 to 7, making this a multiclass classification problem.


### 1.2. Initial Statistical Insights

#### Feature Statistics

The dataset contains 43,500 observations with 9 numerical features and 1 target variable.

Some features show a wide range of values:

- Feature 2:
  - Mean = -0.21
  - Standard deviation = 78.14
  - Minimum = -4821
  - Maximum = 5075

- Feature 6:
  - Mean = 1.30
  - Standard deviation = 179.49
  - Minimum = -13839
  - Maximum = 13148

- Feature 8:
  - Mean = 50.90
  - Standard deviation = 21.46
  - Minimum = -353
  - Maximum = 270

> #### Key Findings
>
>- Several features have large differences between their minimum, maximum, and mean values, indicating possible extreme observations and the need for further investigation during exploratory data analysis.
>
>- Feature scales vary significantly across the dataset, suggesting that feature normalization or standardization may be beneficial before neural network training.
>
>- The target variable contains multiple classes, requiring multiclass classification techniques.

### 1.3. Data Quality Assessment

The dataset was evaluated for potential data quality issues before further analysis and preprocessing.

#### Missing Values

- Total missing values: 0

#### Duplicate Records

- Total duplicate rows: 0

#### Data Types

- All features are numerical (`int64`)
- No categorical encoding is required.


## 2. Exploratory Data Analysis (EDA)

Comprehensive analysis of feature distributions, relationships, and data patterns to understand underlying structures in the dataset.

### Feature 1 - Histogram

<p align="center">
  <img src="assets/feature_1_histogram.png" width="100%">
</p>

### Feature 1 - Boxplot

<p align="center">
  <img src="assets/feature_1_boxplot.png" width="100%">
</p>

>**Key Observations**
>
>**Asymmetrical Distribution (Skewness):**
The histogram displays a right-skewed distribution featuring prominent density peaks around the values of **38** and **55**. From these high-frequency modes, the distribution extends into a long right tail. Because this pronounced positive skew can destabilize gradient-based optimizers during model training, a distribution transformation would be beneficial. Since the feature appears to contain strictly positive values, I recommend applying a **Log Transformation**. This approach is likely to normalize the distribution and stabilize the variance before the data is passed to my model.
>
>**Extreme Outlier Concentration:**
The boxplot indicates a dense, continuous sequence of extreme outliers starting immediately beyond the upper bound defined by $1.5 \times IQR$ (approximately **80**) and reaching up to **125**. Rather than isolated measurement anomalies, these concentrated points align with the secondary clusters visible in the histogram. Therefore, they appear to represent valid observations that I need to preserve to avoid significant information loss.
>
>**Scaling Recommendation:**
I recommend utilizing a **`RobustScaler`** for this feature. Because the data contains a heavy concentration of valid, extreme outliers, a mean-based approach like `StandardScaler` would be heavily distorted. `RobustScaler` relies on the median and the interquartile range (which sits tightly between **38** and **55**), meaning it is highly likely to scale the feature effectively without squashing the variance of my dominant baseline readings.