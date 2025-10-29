"""
Formula 1 Race Prediction Model – Random Forest Analysis (using FastF1 data)
This script loads real F1 race data via FastF1, preprocesses it, trains
RandomForest models (regressor & classifier) to predict outcomes,
evaluates them, and visualises results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import fastf1
from fastf1 import plotting

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score
)

import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("=" * 80)
print("FORMULA 1 RACE PREDICTION MODEL – RANDOM FOREST ANALYSIS")
print("=" * 80)

# Enable cache for FastF1
fastf1.Cache.enable_cache('fastf1_cache')  # Change path as needed
plotting.setup_mpl(color_scheme='fastf1')

# ============================================================================
# STEP 1: DATA LOAD via FastF1
# ============================================================================
print("\n[STEP 1] Loading F1 data via FastF1…")

def load_f1_data(season=2024, n_rounds=5):
    """Load race session data for a given season & number of rounds."""
    records = []
    for rnd in range(1, n_rounds + 1):
        try:
            session = fastf1.get_session(season, rnd, 'R')
            session.load(laps=True, telemetry=False, weather=True)
            print(f"  → Loaded season {season}, round {rnd}")
            
            laps = session.laps  # DataFrame of laps
            results = session.results  # DataFrame of final classification
            
            # For each driver in this race
            for _, row in results.iterrows():
                driver = row['Abbreviation']
                team = row['TeamName']
                grid_pos = row['GridPosition']
                race_pos = row['Position']
                points = row['Points']
                
                # Compute avg lap time in seconds for driver
                driver_laps = laps.pick_driver(driver)
                if driver_laps.empty:
                    continue
                avg_lap_time = driver_laps['LapTime'].dt.total_seconds().mean()
                laps_completed = len(driver_laps)
                
                # pit_stop count – approximate by counting PitOutTime
                pit_outs = driver_laps['PitOutTime'].notna().sum()
                
                # weather: take average air temp (if available)
                if session.weather_data is not None:
                    air_temp = session.weather_data['AirTemp'].mean()
                else:
                    air_temp = np.nan
                
                records.append({
                    'race_id': rnd,
                    'driver': driver,
                    'team': team,
                    'qualifying_position': grid_pos,
                    'grid_position': grid_pos,
                    'race_position': race_pos,
                    'points': points,
                    'laps_completed': laps_completed,
                    'avg_lap_time': avg_lap_time,
                    'pit_stops': pit_outs,
                    'track_name': session.event['EventName'],
                    'weather_air_temp': air_temp,
                    'season': season
                })
        except Exception as e:
            print(f"  ⚠️ Could not load round {rnd}: {e}")
    return pd.DataFrame(records)

df = load_f1_data(season=2024, n_rounds=10)
print(f"\nLoaded {df.shape[0]} records")
print(df.head())
print("\nMissing values:\n", df.isnull().sum())

# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 2] Data Preprocessing…")
print("=" * 80)

df_processed = df.copy()

# 2.1 Handle missing values
print("\n[2.1] Handling Missing Values…")
df_processed['avg_lap_time'].fillna(df_processed['avg_lap_time'].median(), inplace=True)
df_processed['weather_air_temp'].fillna(df_processed['weather_air_temp'].median(), inplace=True)
df_processed['pit_stops'].fillna(df_processed['pit_stops'].median(), inplace=True)
print("Missing values handled via median imputation.")

# 2.2 Encode categorical variables
print("\n[2.2] Encoding Categorical Variables…")
label_encoders = {}
categorical_cols = ['driver', 'team', 'track_name']

for col in categorical_cols:
    le = LabelEncoder()
    df_processed[col + '_encoded'] = le.fit_transform(df_processed[col])
    label_encoders[col] = le
    print(f"  - {col}: {len(le.classes_)} unique values")

# 2.3 Select features & targets
feature_cols = [
    'qualifying_position', 'grid_position', 'laps_completed',
    'avg_lap_time', 'pit_stops', 'weather_air_temp', 'season',
    'driver_encoded', 'team_encoded', 'track_name_encoded'
]
X = df_processed[feature_cols].copy()
y_regression = df_processed['race_position'].copy()
y_classification = (df_processed['race_position'] <= 10).astype(int)

print(f"\nFeatures selected ({len(feature_cols)}): {feature_cols}")
print("Target (Regression): race_position")
print("Target (Classification): top_10_finish (1=Yes,0=No)")

# 2.4 Scale numerical features
print("\n[2.3] Scaling Numerical Features…")
scaler = StandardScaler()
numerical_cols = ['qualifying_position', 'grid_position', 'laps_completed',
                  'avg_lap_time', 'pit_stops', 'weather_air_temp']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
print("Numerical features scaled.")

# 2.5 Split data
print("\n[2.4] Splitting Data (80% train, 20% test)…")
X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X, y_regression, test_size=0.2, random_state=42
)
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X, y_classification, test_size=0.2, random_state=42
)
print(f"Regression → Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
print(f"Classification → Train: {X_clf_train.shape[0]}, Test: {X_clf_test.shape[0]}")

# ============================================================================
# STEP 3: MODEL BUILDING & TRAINING
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 3] Building & Training Random Forest Models…")
print("=" * 80)

# 3.1 Regression
print("\n[3.1] Training RandomForestRegressor…")
rf_regressor = RandomForestRegressor(
    n_estimators=100, max_depth=15,
    min_samples_split=5, min_samples_leaf=2,
    random_state=42, n_jobs=-1
)
rf_regressor.fit(X_train, y_reg_train)
print("Regressor trained.")

# 3.2 Classification
print("\n[3.2] Training RandomForestClassifier…")
rf_classifier = RandomForestClassifier(
    n_estimators=100, max_depth=15,
    min_samples_split=5, min_samples_leaf=2,
    random_state=42, n_jobs=-1
)
rf_classifier.fit(X_clf_train, y_clf_train)
print("Classifier trained.")

# ============================================================================
# STEP 4: EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 4] Model Evaluation…")
print("=" * 80)

# Regression eval
print("\n[4.1] Regression Model Evaluation")
y_reg_pred = rf_regressor.predict(X_test)
mae = mean_absolute_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
r2 = r2_score(y_reg_test, y_reg_pred)
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Classification eval
print("\n[4.2] Classification Model Evaluation")
y_clf_pred = rf_classifier.predict(X_clf_test)
y_clf_pred_proba = rf_classifier.predict_proba(X_clf_test)[:, 1]
accuracy = accuracy_score(y_clf_test, y_clf_pred)
precision = precision_score(y_clf_test, y_clf_pred)
recall = recall_score(y_clf_test, y_clf_pred)
f1 = f1_score(y_clf_test, y_clf_pred)
roc_auc = roc_auc_score(y_clf_test, y_clf_pred_proba)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

print("\n[4.3] Confusion Matrix")
cm = confusion_matrix(y_clf_test, y_clf_pred)
print(f"TN: {cm[0,0]}  FP: {cm[0,1]}")
print(f"FN: {cm[1,0]}  TP: {cm[1,1]}")

# ============================================================================
# STEP 5: Feature Importance
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 5] Feature Importance (Regression Model)…")
print("=" * 80)
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_regressor.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.head(10).to_string(index=False))

# ============================================================================
# STEP 6: Visualisations
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 6] Visualisations…")
print("=" * 80)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

fig = plt.figure(figsize=(18, 14))

# Feature importance bar
ax1 = plt.subplot(3,3,1)
topf = feature_importance.head(10)
ax1.barh(range(len(topf)), topf['importance'], color='steelblue')
ax1.set_yticks(range(len(topf)))
ax1.set_yticklabels(topf['feature'])
ax1.set_xlabel('Importance Score')
ax1.set_title('Top 10 Feature Importances')
ax1.invert_yaxis()

# Predicted vs actual regression
ax2 = plt.subplot(3,3,2)
ax2.scatter(y_reg_test, y_reg_pred, alpha=0.6, color='darkgreen')
ax2.plot([y_reg_test.min(), y_reg_test.max()],
         [y_reg_test.min(), y_reg_test.max()], 'r--', lw=2)
ax2.set_xlabel('Actual Race Position')
ax2.set_ylabel('Predicted Race Position')
ax2.set_title('Predicted vs Actual (Regression)')

# Residuals
ax3 = plt.subplot(3,3,3)
residuals = y_reg_test - y_reg_pred
ax3.scatter(y_reg_pred, residuals, alpha=0.6, color='purple')
ax3.axhline(y=0, color='r', linestyle='--', lw=2)
ax3.set_xlabel('Predicted Values')
ax3.set_ylabel('Residuals')
ax3.set_title('Residuals Plot')

# Confusion matrix heatmap
ax4 = plt.subplot(3,3,4)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4, cbar=False)
ax4.set_xlabel('Predicted')
ax4.set_ylabel('Actual')
ax4.set_title('Confusion Matrix (Top 10)')

# ROC curve
ax5 = plt.subplot(3,3,5)
fpr, tpr, _ = roc_curve(y_clf_test, y_clf_pred_proba)
ax5.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.3f})')
ax5.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
ax5.set_xlabel('False Positive Rate')
ax5.set_ylabel('True Positive Rate')
ax5.set_title('ROC Curve')
ax5.legend(loc='lower right')

# Error distribution
ax6 = plt.subplot(3,3,6)
errors = np.abs(y_reg_test - y_reg_pred)
ax6.hist(errors, bins=20, color='crimson', alpha=0.7, edgecolor='black')
ax6.axvline(mae, color='blue', linestyle='--', lw=2, label=f'MAE = {mae:.3f}')
ax6.set_xlabel('Absolute Error')
ax6.set_ylabel('Frequency')
ax6.set_title('Error Distribution')
ax6.legend()

plt.tight_layout()
plt.savefig('f1_model_analysis_fastf1.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'f1_model_analysis_fastf1.png'")
plt.show()

# ============================================================================
# STEP 7: Hyperparameter Tuning (optional)
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 7] Hyperparameter Tuning – GridSearchCV")
print("=" * 80)

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 15, 20],
    'min_samples_split': [3, 5, 7]
}

print("\nTuning RandomForestRegressor with param grid:", param_grid)
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_reg_train)
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV R² Score: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
r2_best = r2_score(y_reg_test, y_pred_best)
mae_best = mean_absolute_error(y_reg_test, y_pred_best)
print(f"\nBest Model Test R²: {r2_best:.4f}")
print(f"Best Model Test MAE: {mae_best:.4f}")

# ============================================================================
# STEP 8: Predictions for Upcoming Races (example)
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 8] Predictions for Upcoming Races…")
print("=" * 80)

# Create synthetic upcoming race data (replace with real qualification data when available)
upcoming_data = {
    'qualifying_position': [3, 5, 1, 8, 2],
    'grid_position': [3, 5, 1, 8, 2],
    'laps_completed': [57, 57, 57, 56, 57],
    'avg_lap_time': [91.8, 92.5, 90.9, 93.4, 92.1],
    'pit_stops': [2,2,2,3,2],
    'weather_air_temp': [30,30,30,30,30],
    'season': [2024,2024,2024,2024,2024],
    'driver_encoded': [0,1,2,3,4],
    'team_encoded': [0,1,2,3,4],
    'track_name_encoded': [0,1,2,3,4]
}

X_upcoming = pd.DataFrame(upcoming_data)
X_upcoming[numerical_cols] = scaler.transform(X_upcoming[numerical_cols])

pred_reg = rf_regressor.predict(X_upcoming)
pred_clf = rf_classifier.predict(X_upcoming)
pred_proba = rf_classifier.predict_proba(X_upcoming)[:,1]

print("\nUpcoming Race Predictions:")
print("-" * 80)
for i,(pos, top10, prob) in enumerate(zip(pred_reg, pred_clf, pred_proba)):
    print(f"Driver {i+1}: Predicted Finish: {pos:.1f}, Top 10: {'Yes' if top10==1 else 'No'} (Prob: {prob:.2%})")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print(f"\nKey Findings:")
print(f"  - Regression Model R² Score: {r2:.4f}")
print(f"  - Classification Model Accuracy: {accuracy:.4f}")
print(f"  - Most Important Feature: {feature_importance.iloc[0]['feature']}")
print("\nNext Steps:")
print("  - Expand data: include more rounds/seasons")
print("  - Use richer features: qualifying times, sector times, weather metrics")
print("  - Deploy model as service or dashboard")
print("=" * 80)
