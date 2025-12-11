
import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("HR Employee Attrition.csv")
print(df.shape)
print('Data shape:', df.shape)
df.head()
print(df.info())
print(df.describe(include='all').T)
print(df.isnull().sum())
print(df['Attrition'].value_counts())
data = df.copy()
to_drop = ['EmployeeCount','Over18','StandardHours','EmployeeNumber'] if set(['EmployeeCount','Over18','StandardHours','EmployeeNumber']).issubset(data.columns) else []
data.drop(columns=to_drop, inplace=True, errors='ignore')
le = LabelEncoder()
data['Attrition_flag'] = le.fit_transform(data['Attrition'])  # Yes=1, No=0 (check mapping)
print('Attrition mapping:', dict(zip(le.classes_, le.transform(le.classes_))))
y = data['Attrition_flag']
X = data.drop(columns=['Attrition','Attrition_flag'])
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
len(cat_cols), len(num_cols)
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
print('Encoded shape:', X_encoded.shape)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)
print(X_train.shape, X_test.shape)
scaler = StandardScaler()
X_train_loc = X_train.copy()
X_test_loc = X_test.copy()
if len([c for c in num_cols if c in X_train_loc.columns])>0:
    X_train_loc[num_cols] = scaler.fit_transform(X_train_loc[num_cols])
    X_test_loc[num_cols] = scaler.transform(X_test_loc[num_cols])
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train_loc, y_train)
y_pred = rf.predict(X_test_loc)
y_proba = rf.predict_proba(X_test_loc)[:,1]
print(classification_report(y_test, y_pred))
print('ROC AUC:', roc_auc_score(y_test, y_proba))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
importances = pd.Series(rf.feature_importances_, index=X_train_loc.columns).sort_values(ascending=False)
print(importances.head(20))
plt.figure(figsize=(8,6))
importances.head(15).plot(kind='barh')
plt.gca().invert_yaxis()
plt.title('Top 15 feature importances (Random Forest)')
plt.show()
os.makedirs('artifacts', exist_ok=True)
joblib.dump(rf, 'artifacts/attrition_rf_model.pkl')
joblib.dump(scaler, 'artifacts/feature_scaler.pkl')
joblib.dump(X_encoded.columns.tolist(), 'artifacts/feature_columns.pkl')
print('Saved model and artifacts to /mnt/data/artifacts')