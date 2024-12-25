# results_module.py
import numpy as np
import pandas as pd
import joblib
import os

# Load star data
def load_star_data(path):
    return pd.read_csv(path, sep='\t')

# Load weights and hyperparameters
def load_weights_and_hyperparams(path):
    return joblib.load(path)

# Load saved models
def load_saved_models(model_save_path, targets):
    models = {}
    for target in targets:
        model_path = model_save_path.format(target=target)
        models[target] = joblib.load(model_path)
    return models

# Generate Gaussian realizations
def generate_realizations(row, n_realizations=10000):
    teff_realizations = np.random.normal(row['Teff'], row['err_Teff'], n_realizations)
    feh_realizations = np.random.normal(row['FeH'], row['err_FeH'], n_realizations)
    l_realizations = np.random.normal(row['lum'], row['err_lum'], n_realizations)
    return pd.DataFrame({'Teff': teff_realizations, 'FeH': feh_realizations, 'L': l_realizations})

# Calculate weighted prediction
def calculate_final_prediction(model_predictions, weights, decimals):
    stacked_predictions = np.dot(model_predictions, weights)
    median = np.round(np.median(stacked_predictions), decimals)
    lower_bound = np.round(np.percentile(stacked_predictions, 16), decimals)
    upper_bound = np.round(np.percentile(stacked_predictions, 84), decimals)
    upper_offset = np.round(upper_bound - median, decimals)
    lower_offset = np.round(lower_bound - median, decimals)
    return stacked_predictions, median, lower_bound, upper_bound,upper_offset, lower_offset

# Save prediction results to file
def save_predictions(output_dir, star_id, target, predictions):
    file_path = os.path.join(output_dir, f'{star_id}_{target}_predictions.npy')
    np.save(file_path, predictions)

# Summarize predictions for each star in the desired structure
def summarize_predictions(summary_list, star_id, mass, ub_mass, lb_mass, radius, ub_radius, lb_radius, age, ub_age, lb_age):
    summary_list.append({
        'object_name': star_id,
        'mass': mass,
        'ub_mass': ub_mass,
        'lb_mass': lb_mass,
        'radius': radius,
        'ub_radius': ub_radius,
        'lb_radius': lb_radius,
        'age': age,
        'ub_age': ub_age,
        'lb_age': lb_age
    })

