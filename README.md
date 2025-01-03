![logo](logos/black_bkg_logo.png)

This repository contains code for characterizing stars, with specific focus on parameters like radius, mass, and age using a stacking-like ML approach 

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Processing](#data-processing)
- [Model Training](#model-training)
- [Prediction](#prediction)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project involves training various machine learning algorithms on stellar models to predict stellar parameters from observational data. It supports ensemble algorithms like ExtraTreesRegressor, XGBoost, and more. Additionally, the project includes methods for generating noisy datasets and making predictions with uncertainty quantification.

## Features

- **Preprocessing and Transformation**: Transformations such as logarithmic scaling and filtering on astrophysical data.
- **Ensemble Modeling**: Use of various regressors (e.g., Extra Trees, XGBoost) to train models on stellar features.
- **Cross-Validation**: K-Fold cross-validation for model evaluation.
- **Uncertainty Estimation**: Generate noisy data to provide predictions with uncertainty.
- **Prediction for Real Data**: Load real observational data, make predictions, and calculate statistical metrics.
- **Visualization**: Generate and save histograms for result analysis.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Kamulali/MAISTEP-article.git
    cd MAISTEP-article
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv myenv
    source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
   ```bash
   cd MAISTEP-article/code
   python3 main_code_v1.py
   ```

1. **Loading and Preprocessing the Data**:
   The `load_dataset()` and `preprocess_transformations()` functions are used to load the data from a file and apply necessary transformations.

    Example:
    ```python
    from main import load_dataset, preprocess_transformations
    
    file_path = "path/to/your/datafile.txt"
    dataset = load_dataset(file_path)
    transformed_data = preprocess_transformations(dataset, apply_function)
    ```
## Input Data Formats

### Training Data File
The stellar grid for training the ML algorithms should be in a tab-separated `.csv` format. It includes both **features** (independent variables) and **target labels** (dependent variables).

- **Training Features**:
  - `Teff`: Effective temperature (K).
  - `[Fe/H]`: Metallicity (dex).
  - `L`: Luminosity (Lsolar).

- **Training Target Labels**:
  - `mass`: (Msolar)
  - `radius`: (Rsolar)
  - `age`: (Gyr)
See an example format in /data/APOKASC_constraints_Baye_2019.csv 

### Observation Data File
The observation data file must include object names (col_1) and corresponding observed values (col_2,col_4,col_6,...) and uncertainties (col_3,col_5,col_7,...) for the features (`Teff`, `[Fe/H]`, `L`). And should be tab-separated.

#### Sample format for observation data:
##### Object Name    Teff    Teff_error    [Fe/H]    [Fe/H]_error    L    L_error
##### Sun            5777    50            0.0       0.1           1.0  0.05


2. **Training the Models**:
   Train models for each target (mass, radius, age) using the `train_base_models()` function.

    Example:
    ```python
    train_base_models(X_train, y_train, y_test, target)
    ```

3. **Prediction on New Data**:
   Make predictions on noisy or real observational data using trained models.

    Example:
    ```python
    predict_noisy_data(base_models_dict, meta_model_coefficients_dict, noisy_data_dfs, scaler, obj_names)
    ```
4. **Results Visualization**:
   Save and visualize results using histograms for predicted parameters.

    Example:
    ```python
    analyze_data(directory_path, results_directory)
    ```

## Data Processing

- **Preprocessing**: Dataset transformations and filtering are handled in `preprocess_transformations()` and `preprocess_filters()`.
- **Scaling**: The features are normalized using `RobustScaler` to make them more robust to outliers.

## Model Training

- The project supports multiple machine learning models (Extra Trees, XGBoost, etc.) trained with 10-fold cross-validation.
- Predictions from multiple models are combined using non-negative least squares (NNLS) to provide a final prediction.

## Prediction

- Predictions are generated using base models and combined through weighting.
- Supports creating random noise data from  observations' standard deviations, subsequently providing errors on each target parameter.

## Results

- Predicted results are stored as `.txt` files for each object.
- Summary statistics (median, upper and lower bounds) are calculated for mass, radius, and age.
- Histograms are produced for visualisation.

### Output Data

#### Sample Format:
##### Object Name    mass   mass_upper_bound  mass_lower_bound  radius   radius_upper_bound  radius_lower_bound  age   age_upper_bound  age_lower_bound
##### Sun            1.0    0.02              -0.02             1.0      0.01                -0.01               4.6   0.1              -0.1


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
