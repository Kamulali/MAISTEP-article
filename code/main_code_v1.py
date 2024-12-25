import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from scipy.optimize import nnls
from sklearn.metrics import mean_squared_error
import optuna
import joblib
import os
# custom modules 
import results_module
import plotting_module


def load_dataset(file_path):
    """Load the dataset from the given file path."""
    print("Loading dataset...")
    return pd.read_csv(file_path, low_memory=False, sep='\t')
    
def preprocess_transformations(dataset, apply_function):
    """Apply specified transformations to the dataset."""
    print("Applying transformations to dataset...")
    for column_name, transform_name in apply_function.items():
        if column_name in dataset.columns:
            if transform_name == "del":
                dataset.drop(columns=column_name, inplace=True)
                print(f"Dropped column: {column_name}")
            elif transform_name == "log":
                dataset[column_name] = np.log(dataset[column_name])
                print(f"Applied log transform to column: {column_name}")
    return dataset

def preprocess_filters(dataset):
    """Filter dataset based on specified conditions."""
    print("Applying filters to dataset...")
    return dataset[(dataset['Age'] >= 0) & (dataset['Age'] <= 14)]

def objective(trial, model_name, X_train, y_train):
    """Objective function for Optuna hyperparameter tuning."""
    if model_name     == "CatBoost":
        depth         = trial.suggest_int('depth', 1,5 )
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 1, log=True)
        iterations    = trial.suggest_int('iterations', 500, 1500)
        model         = CatBoostRegressor(iterations=iterations, depth=depth, learning_rate=learning_rate, silent=True)
    elif model_name   == "XT":
        max_depth     = trial.suggest_int('max_depth', 5, 20)
        model         = ExtraTreesRegressor(max_depth=max_depth, random_state=42, n_jobs=19)
    elif model_name   == "XGBoost":
        max_depth     = trial.suggest_int('max_depth', 3, 6)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
        model         = XGBRegressor(n_estimators=1000, max_depth=max_depth, learning_rate=learning_rate, random_state=42, n_jobs=19)
    elif model_name   == "RF":
        n_estimators  = trial.suggest_int('n_estimators', 50, 300)
        max_depth     = trial.suggest_int('max_depth', 5, 20)
        model         = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=19)
    
    model.fit(X_train, y_train)
    y_pred            = model.predict(X_train)
    rmse              = np.sqrt(mean_squared_error(y_train, y_pred))
    return rmse

# Load and preprocess the dataset / path to the training data
file_path = '/home/kamju/MAISTEP-article/data/MAISTEP_uniform_sampled_grid_file.txt'
dataset   = load_dataset(file_path)

# Specify transformations if needed, and apply them
apply_function = {
    # "lum": "log",
    # "Radius": "log",
    # "Teff": "log",
    # "Xc": "log",
    # "Xs": "log",
    # "Zs": "log",
    # "Age_adim": "del"
}
dataset = preprocess_transformations(dataset, apply_function)
dataset = preprocess_filters(dataset)

# Define features and target parameters
features = ['Teff', 'FeH', 'L']
targets  = ['Radius', 'Mass', 'Age']
X        = dataset[features]

# Initialize cross-validation strategy
kf       = KFold(n_splits=10, shuffle=True, random_state=42)

# Define the base models
base_models = [
    ("CatBoost", CatBoostRegressor(random_state=42, silent=True)),
    ("XT", ExtraTreesRegressor(random_state=42, n_jobs=-1)),
    ("XGBoost", XGBRegressor(n_estimators=1000, random_state=42, n_jobs=18)),
    ("RF", RandomForestRegressor(random_state=42, n_jobs=18)),
]

# Dictionary to store stacking weights and hyperparameters
weights_and_hyperparams = {}

for target in targets:
    print(f"\nProcessing target: {target}")
    y = dataset[target]

    # Split data into cross-validation and test sets
    X_cv, X_test, y_cv, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_cv = y_cv.squeeze()

    # Initialize placeholder for meta-features and best hyperparameters
    meta_features = np.zeros((X_cv.shape[0], len(base_models)))
    best_params_dict = {}

    for i, (name, _) in enumerate(base_models):
        print(f"Hyperparameter tuning for model: {name}")

        # Optuna study for hyperparameter tuning
        study = optuna.create_study()
        study.optimize(lambda trial: objective(trial, name, X_cv, y_cv), n_trials=50)
        
        # Retrieve and store the best hyperparameters
        best_params = study.best_params
        best_params_dict[name] = best_params
        print(f"Best hyperparameters for {name}: {best_params}")

        # Initialize the model with the best parameters
        if name   == "CatBoost":
            model = CatBoostRegressor(iterations=best_params['iterations'], depth=best_params['depth'], 
                                      learning_rate=best_params['learning_rate'], silent=True)
        elif name == "XT":
            model = ExtraTreesRegressor(max_depth=best_params['max_depth'], random_state=42, n_jobs=19)
        elif name == "XGBoost":
            model = XGBRegressor(n_estimators=1500, max_depth=best_params['max_depth'], 
                                 learning_rate=best_params['learning_rate'], random_state=42, n_jobs=19)
        elif name == "RF":
            model = RandomForestRegressor(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], 
                                          random_state=42, n_jobs=19)

        # Train and collect predictions for each fold in cross-validation
        print("Starting cross-validation for meta-feature generation...")
        for train_index, val_index in kf.split(X_cv):
            X_train, X_val = X_cv.iloc[train_index], X_cv.iloc[val_index]
            y_train, y_val = y_cv.iloc[train_index], y_cv.iloc[val_index]
            model.fit(X_train, y_train)
            meta_features[val_index, i] = model.predict(X_val)

    # Calculate stacking weights using NNLS
    print("Computing stacking weights using NNLS...")
    nnls_weights, _ = nnls(meta_features, y_cv.values.flatten())
    nnls_weights /= np.sum(nnls_weights)  # Normalize weights
    print(f"Stacking weights for {target}: {nnls_weights}")

    # Save the weights and hyperparameters for this target
    weights_and_hyperparams[target] = {
        "weights": nnls_weights,
        "hyperparameters": best_params_dict
    }

    # Retrain models on the full dataset with the best hyperparameters and save the trained model
    trained_models = {}
    for name, _ in base_models:
        print(f"Training full dataset for {target} with model: {name}")
        if name   == "CatBoost":
            model = CatBoostRegressor(**best_params_dict[name], silent=True)
        elif name == "XT":
            model = ExtraTreesRegressor(**best_params_dict[name], random_state=42, n_jobs=19)
        elif name == "XGBoost":
            model = XGBRegressor(**best_params_dict[name], random_state=42, n_jobs=19)
        elif name == "RF":
            model = RandomForestRegressor(**best_params_dict[name], random_state=42, n_jobs=19)
        
        model.fit(X, y)
        trained_models[name] = model

    # Save the trained models for the current target
    saved_models_path = f"/home/kamju/MAISTEP-article/weights_hp_models_dir/trained_models_{target}.joblib"
    joblib.dump(trained_models, saved_models_path)
    print(f"Trained models for {target} saved to {saved_models_path}")

# Save weights and hyperparameters for all targets to disk
save_path = '/home/kamju/MAISTEP-article/weights_hp_models_dir/weights_hyperparams.pkl'
joblib.dump(weights_and_hyperparams, save_path)
print(f"\nWeights and hyperparameters saved successfully to {save_path}.")

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@ Load and display the saved model details and hyperparameters for a specified target @@@@@@@@@@@@@


def load_and_display_model_details(target):
    # Load the saved model
    saved_models_path = f"/home/kamju/MAISTEP-article/weights_hp_models_dir/trained_models_{target}.joblib"
    saved_models = joblib.load(saved_models_path)
    
    # Load the saved weights and hyperparameters
    save_path = '/home/kamju/MAISTEP-article/weights_hp_models_dir/weights_hyperparams.pkl'
    weights_and_hyperparams = joblib.load(save_path)

    # Print model parameters
    print(f"\nDetails of saved models for target: {target}")
    for model_name, model in saved_models.items():
        print(f"{model_name} parameters: {model.get_params()}")
    
    # Print weights and hyperparameters
    print(f"\nWeights and hyperparameters for target {target}:")
    if target in weights_and_hyperparams:
        weights = weights_and_hyperparams[target]['weights']
        hyperparams = weights_and_hyperparams[target]['hyperparameters']
        print(f"Weights: {weights}")
        print(f"Hyperparameters: {hyperparams}")
    else:
        print(f"No weights and hyperparameters found for target: {target}")

# Example usage
load_and_display_model_details('Radius')  # Replace 'Radius' with 'Mass' or 'Age' as needed

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ test run for the Sun @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Define the star features as a DataFrame
star_features = pd.DataFrame({
                              'Teff': [5777],
                               'FeH': [0.0],
                                 'L': [1]
                             })

# Load the weights and hyperparameters
weights_and_hyperparams_path = '/home/kamju/MAISTEP-article/weights_hp_models_dir/weights_hyperparams.pkl'
weights_and_hyperparams = joblib.load(weights_and_hyperparams_path)

# Load the saved models
saved_models_path = "/home/kamju/MAISTEP-article/weights_hp_models_dir/trained_models_{target}.joblib"
#targets           = ['Radius', 'Mass', 'Age']  # Specify your target parameters
predictions       = {}

for target in targets:
    model_path  = saved_models_path.format(target=target)
    saved_model = joblib.load(model_path)
    
    # Use the model to predict the star parameters
    print(f"\nPredicting {target} using saved model...")
    model_predictions = []
    
    for model_name, model in saved_model.items():
        prediction = model.predict(star_features)
        model_predictions.append(prediction[0])  # Store the prediction
        print(f"{model_name} predicted {target}: {prediction[0]}")
    
    # Combine predictions using the saved weights
    weights = weights_and_hyperparams[target]['weights']
    combined_prediction = np.dot(weights, model_predictions)  # Weighted sum
    print(f"Combined prediction for {target}: {combined_prediction}")
    
    predictions[target] = {
        'individual_predictions': model_predictions,
        'combined_prediction': combined_prediction
    }

# Display all predictions
print("\nAll predictions:")
for target, preds in predictions.items():
    print(f"\nPredictions for {target}:")
    for model_name, value in zip(saved_model.keys(), preds['individual_predictions']):
        print(f"{model_name}: {value}")
    print(f"Combined prediction: {preds['combined_prediction']}")

# generating the distributions & making plots

# Define paths
constraints_file_path        = '/home/kamju/MAISTEP-article/data/APOKASC_constraints_Baye_2019.csv'
weights_and_hyperparams_path = '/home/kamju/MAISTEP-article/weights_hp_models_dir/weights_hyperparams.pkl'
saved_models_path            = "/home/kamju/MAISTEP-article/weights_hp_models_dir/trained_models_{target}.joblib"
results_output_dir           = '/home/kamju/MAISTEP-article/results'
results_plot_dir             = '/home/kamju/MAISTEP-article/results/plots'
#targets = ['Radius', 'Mass', 'Age']

# Load data, models, and weights
star_data               = results_module.load_star_data(constraints_file_path)
weights_and_hyperparams = results_module.load_weights_and_hyperparams(weights_and_hyperparams_path)
saved_models            = results_module.load_saved_models(saved_models_path, targets)

# Initialize summary list
summary_list = []
os.makedirs(results_output_dir, exist_ok=True)
os.makedirs(results_plot_dir, exist_ok=True)

# Iterate over each star
for index, row in star_data.iterrows():
    star_id = row['target_id']
    print(f"Processing star: {star_id} (Index {index + 1}/{len(star_data)})")

    realizations_df = results_module.generate_realizations(row)
    
    # Temporary storage for each target's summary values
    summary_values = {}

    for target in targets:
        stacked_predictions = []
        for model_name, model in saved_models[target].items():
            model_predictions = model.predict(realizations_df)
            stacked_predictions.append(model_predictions)

        model_predictions = np.column_stack(stacked_predictions)
        weights = weights_and_hyperparams[target]['weights']
        decimals = 3 if target == 'Radius' else 2
        unit = 'R$_{\odot}$' if target == 'Radius' else 'M$_{\odot}$' if target == 'Mass' else 'Gyr'
        
        # Calculate final predictions
        predictions, median, lower_bound, upper_bound,upper_offset, lower_offset = results_module.calculate_final_prediction(
            model_predictions, weights, decimals
        )
        
        # Save individual predictions
        results_module.save_predictions(results_output_dir, star_id, target, predictions)
        
        # Plot predictions and save
        plotting_module.plot_predictions(
            star_id, target, predictions, median, lower_bound, upper_bound, decimals, unit, results_plot_dir
        )
        
        # Store summary values for the current target
        summary_values[target] = {
            'median': median,
            'ub': upper_offset,
            'lb': lower_offset
        }
    
    # Append summary for all targets
    results_module.summarize_predictions(
        summary_list,
        star_id,
        summary_values['Mass']['median'],
        summary_values['Mass']['ub'],
        summary_values['Mass']['lb'],
        summary_values['Radius']['median'],
        summary_values['Radius']['ub'],
        summary_values['Radius']['lb'],
        summary_values['Age']['median'],
        summary_values['Age']['ub'],
        summary_values['Age']['lb']
    )

# Save summary data
summary_df = pd.DataFrame(summary_list)
summary_file_path = os.path.join(results_output_dir, 'summary_predictions.csv')
summary_df.to_csv(summary_file_path, index=False, sep='\t')
print(f"Summary predictions saved to {summary_file_path}")

