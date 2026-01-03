import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib
import os

if not os.path.exists('models'): os.makedirs('models')
if not os.path.exists('results/plots'): os.makedirs('results/plots')

df = pd.read_csv('data/processed/cleaned_data.csv')

X = df.drop('TARGET_CHURN', axis=1)
y = df['TARGET_CHURN']

model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, eval_metric='logloss')
model.fit(X, y)

joblib.dump(model, 'models/model.pkl')
joblib.dump(X.columns.tolist(), 'models/columns.pkl')

fig, ax = plt.subplots(figsize=(10, 8))
xgb.plot_importance(model, max_num_features=10, ax=ax)
plt.title('Feature Importance')
plt.subplots_adjust(left=0.3)
plt.savefig('results/plots/feature_importance.png', bbox_inches='tight')
print("Model Training Complete.")