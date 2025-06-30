# Weather Prediction Using HMM and GRU

This project focuses on predicting daily weather types (e.g., sun, rain, snow) in Seattle using sequential machine learning models:
- **Gaussian Hidden Markov Models (HMM)** for unsupervised temporal pattern learning.
- **Gated Recurrent Unit (GRU)** neural networks for sequence modeling using deep learning.

The project includes comprehensive **data preprocessing**, **feature engineering**, and **hyperparameter optimization** using:
- `Optuna`
- `GridSearchCV`
- `KerasTuner`

---

## Dataset

The dataset used is `seattle-weather.csv`, which contains:
- Daily measurements from 2012 onwards.
- Features: `date`, `precipitation`, `temp_max`, `temp_min`, `wind`, and `weather` (target).

### Sample:
| date       | precipitation | temp_max | temp_min | wind | weather  |
|------------|----------------|----------|----------|------|----------|
| 2012-01-01 | 0.0            | 12.8     | 5.0      | 4.7  | drizzle  |
| 2012-01-02 | 10.9           | 10.6     | 2.8      | 4.5  | rain     |

---

## Preprocessing

The following feature engineering and preprocessing steps were applied:
- **Label Encoding** for `weather`:
  - `{'sun': 0, 'rain': 1, 'snow': 2, 'drizzle': 3, 'fog': 4}`
- **Season Encoding** based on `date`:
  - Winter: 0, Spring: 1, Summer: 2, Autumn: 3
- **Temperature Binning** from `temp_min`
- **Wind Binning** from `wind`
- **Binary precipitation**: `0` (dry), `1` (wet)
- Final features used:
  - `Encode Season`, `Encode Temp`, `Encode Precipitation`, `Encode Wind`, `Encode Weather`

---

## Models and Tuning

### 1. **Hidden Markov Model (HMM)**
- Implemented using `hmmlearn.GaussianHMM`
- Trained on sequences of past days' encoded weather features
- Accuracy is calculated based on prediction of next day's weather class

#### Tuning:
- **Optuna**: Search space included `n_components`, `covariance_type`, `n_iter`
- **GridSearchCV**: Exhaustive tuning of 60+ parameter combinations
- **Randomized Search**: Additional search for `num_previous_days` (sequence length)

---

### 2. **GRU Neural Network**
- Implemented using `TensorFlow` and `Keras`
- Sequence-to-label model predicting next day's weather from past sequence
- Uses `SparseCategoricalCrossentropy` as loss function

#### Tuning:
- **KerasTuner (RandomSearch)**:
  - Tuned: `GRU units`, `Dropout rate`, `Dense layer size`, `Learning rate`
- Final model trained for 20 epochs

---

## Results

| Model | Accuracy (Test Set) |
|-------|----------------------|
| GRU   | **`67%`** |
| HMM   | **`87%`** |

Bar chart comparing both models was generated using `Matplotlib`.

---

## Visualizations

- **Weather distribution**
- **Precipitation vs. weather type**
- **Model performance vs. hyperparameters**
- Accuracy vs. depth / C / components (where applicable)

---

## Technologies Used

- `Python`, `NumPy`, `Pandas`, `Matplotlib`
- `scikit-learn`, `hmmlearn`, `keras`, `tensorflow`
- `Optuna`, `KerasTuner`, `GridSearchCV`

---

## How to Run

1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib scikit-learn hmmlearn optuna keras keras-tuner tensorflow
