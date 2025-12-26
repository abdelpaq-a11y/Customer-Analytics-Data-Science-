
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

"""# Load Dataset"""

df = pd.read_csv('Customers_Fakedata.csv')

df

"""# Get more info about data"""

print(df.describe(include='all').T)

df.head(30)

df.tail(30)

df.shape

df.info()

"""# Check Missing Values"""

df.isnull().sum()

"""# Check Duplicates"""

df.duplicated().sum()

"""# Columns to be cleaned
# Age :
has 520 missing value .
# Gender :
has 273 missing value , Not this repeated .
# Phone :
has 1078 missing value , Should change DT to object DT .  
# PurchaseDate :
Should change DT to Date DT .
# Unnamed :
Drop this Column .
# Gender :
has 273 missing value , Not this repeated .
# ProductCategory :
has 577 missing value .   
# Rating :
has 329 missing value .
# PurchaseAmount :
has 101 missing value .

# Data Cleaning

**Solve fault Dt**
"""

df['Phone'] = df['Phone'].astype('object')
df['PurchaseDate'] = df['PurchaseDate'].replace({'32/13/2020':'2020-12-30'})
df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], format='mixed', dayfirst=False, errors='coerce')
df['PurchaseDate'].isnull().sum()
df['PurchaseDate'].describe()
df.info()

"""**Drop Cloumns**"""

df.drop(columns= ['Unnamed','  Gender  '], inplace=True, errors='ignore')
df.head()

"""**Solve Duplication problem**"""

df.drop_duplicates(inplace=True)
df.duplicated().sum()

"""**Handel Phone Column**"""

if 'Phone' in df.columns:
    df['Phone'].unique()
    df['Phone'].value_counts()
df.drop(columns='Phone', inplace=True, errors='ignore')

df.head(10)

"""**Handel Gender Column**"""

df['Gender'].nunique()
df['Gender'].unique()
df['Gender'].value_counts()
df['Gender'] = df['Gender'].str.lower()
df['Gender'].value_counts()
df['Gender'].replace({'m':'male','f':'female'},inplace=True)
df['Gender'].unique()
df['Gender'].value_counts()
df['Gender'].fillna('male',inplace=True)
df['Gender'].value_counts()

df['Gender'].isnull().sum()

df.describe(include='O').T

"""**Handel Age Column**"""

df['Age'].unique()

df['Age'].min()

df['Age'].max()

sns.histplot(df,x='Age')
plt.show()

"""**Handel Outliers**"""

df['Age'].replace({200.:20,-1.:15.},inplace=True)

df['Age'].describe()

sns.histplot(df,x='Age',kde=True)
plt.show()

df['Age'].fillna(df['Age'].median(),inplace=True)

df['Age'].isnull().sum()

df['Age'].describe()

sns.histplot(df,x='Age',kde=True)
plt.show()

"""**Handel Product Category Column**"""

df['ProductCategory'].value_counts()

df['ProductCategory'].isnull().sum()

df['ProductCategory'].fillna('Other',inplace=True)
df['ProductCategory'].value_counts()

"""**Handel Rating Column**"""

df['Rating'].value_counts()

df['Rating'].replace({10.0:5.0},inplace=True)
df['Rating'].fillna(3.0,inplace=True)
df['Rating'].unique()

"""**Handel Purchase Amount Column**"""

df['PurchaseAmount'].describe()

sns.histplot(df,x='PurchaseAmount',kde=True)
plt.show()

df['PurchaseAmount'].fillna(df['PurchaseAmount'].mean(),inplace=True)
df['PurchaseAmount'].isnull().sum()

sns.histplot(df,x='PurchaseAmount',kde=True)
plt.show()

"""**Check Cleaning**"""

df.info()

df.isnull().sum()

df.head(30)

"""# Encoding + Scaling"""

# Identify numeric + categorical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

# ColumnTransformer (High Performance)
ct = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), cat_cols)
    ],
    remainder='drop'
)

# Transform
X = ct.fit_transform(df)

# Extract encoded column names
num_features = ct.named_transformers_['num'].get_feature_names_out(num_cols)
cat_features = ct.named_transformers_['cat'].get_feature_names_out(cat_cols)
all_features = list(num_features) + list(cat_features)

# Convert sparse → DataFrame
df_encoded = pd.DataFrame.sparse.from_spmatrix(X, columns=all_features)

df_encoded.head()

"""# Save New File"""

df.to_csv('Customers_Fakedata_Cleaned.csv')

"""# Exploratory Data Analysis (EDA)

**Top 10 PurchaseAmount**
"""

df.sort_values(by= 'PurchaseAmount',ascending=False).head(10)

"""**Top Sold Product Category**"""

df['ProductCategory'].value_counts()

"""**Total Slaes**"""

total = round(df['PurchaseAmount'].sum(),1)
formatt = f"${total:,.1f}"
print(formatt)

"""**Avg Sales**"""

Avg = round(df['PurchaseAmount'].mean(),1)
formatt = f"${Avg}"
print(formatt)

"""# What is the most Customer gender segment ?"""

Gender_Count = df['Gender'].value_counts()
Gender_Count

sns.countplot(df,x="Gender",label=Gender_Count)
plt.show()

plt.figure(figsize=(9,7))
plt.pie(Gender_Count,labels=Gender_Count.index,autopct='%1.1f%%')
plt.show()

sns.barplot(df,x='Gender',y='PurchaseAmount')
plt.show()

"""# What is the month we in generates most Sales ?"""

df['Month'] = df['PurchaseDate'].dt.month
round(df.groupby('Month')['PurchaseAmount'].sum(),2).sort_values(ascending =False)

round(df.groupby('Month')['PurchaseAmount'].sum(),2).sort_index().plot(figsize=(15,5),linestyle='--',linewidth=4)
plt.ylabel('Purchase Amount')
plt.xlabel('Month')
plt.title('Monthly Sales',fontsize=24)
plt.show()

"""**Product Category Distribution**"""

plt.figure(figsize=(12,6))
sns.countplot(data=df, x='ProductCategory', order=df['ProductCategory'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Product Category Distribution')
plt.show()

"""**Age vs PurchaseAmount**"""

plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='Age', y='PurchaseAmount', hue='Gender')
plt.title('Correlation between Age and Purchase Amount')
plt.show()

"""**Heatmap Correlation**"""

plt.figure(figsize=(8,6))
sns.heatmap(df[['Age','Rating','PurchaseAmount']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

"""**Purchases per Month**"""

plt.figure(figsize=(12,5))
sns.countplot(data=df, x='Month')
plt.title('Purchase Count by Month')
plt.show()

"""**Scatter Matrix**"""

sns.pairplot(df[['Age','Rating','PurchaseAmount','Gender']], hue='Gender')
plt.show()

"""# Imports"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib

"""# Load & Clean Data (مختصر – بدون EDA)"""

df = pd.read_csv('Customers_Fakedata.csv')

# Fix datatypes
df['Phone'] = df['Phone'].astype('object')
df['PurchaseDate'] = df['PurchaseDate'].replace({'32/13/2020': '2020-12-30'})
df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce')

# Drop useless columns
df.drop(columns=['Unnamed', '  Gender  ', 'Phone'], inplace=True, errors='ignore')

# Remove duplicates
df.drop_duplicates(inplace=True)

# Gender
df['Gender'] = df['Gender'].str.lower()
df['Gender'].replace({'m': 'male', 'f': 'female'}, inplace=True)
df['Gender'].fillna('male', inplace=True)

# Age
df['Age'].replace({200.: 20, -1.: 15.}, inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)

# ProductCategory
df['ProductCategory'].fillna('Other', inplace=True)

# Rating
df['Rating'].replace({10.0: 5.0}, inplace=True)
df['Rating'].fillna(3.0, inplace=True)

# PurchaseAmount
df['PurchaseAmount'].fillna(df['PurchaseAmount'].mean(), inplace=True)

# Month Feature
df['Month'] = df['PurchaseDate'].dt.month

"""# Target Variable"""

df['is_electronics'] = np.where(
    df['ProductCategory'] == 'Electronics', 1, 0
)

"""# Features Selection"""

num_features = ['Age', 'PurchaseAmount', 'Rating', 'Month']
cat_features = ['Gender']

X = df[num_features + cat_features]
y = df['is_electronics']

"""# Train / Test Split"""

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

"""# Preprocessing + Model (Pipeline)"""

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ]
)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=7,
    max_features='log2',
    bootstrap=False,
    random_state=42
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', rf_model)
])

"""# Train & Evaluate"""

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Final Accuracy: {accuracy:.4f}")

"""Confusion Matrix"""

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot(cmap='Blues')
plt.title("Confusion Matrix - RandomForest (Unweighted)")
plt.show()

"""# Save Model"""
import joblib

# Save Model
import joblib

joblib.dump(pipeline, "electronics_rf_pipeline.joblib")



"""# Prediction Function (API / Deployment Ready)"""

def predict_category(new_data: dict) -> str:
    model = joblib.load('electronics_rf_pipeline.joblib')
    df_new = pd.DataFrame([new_data])
    pred = model.predict(df_new)[0]
    return 'Electronics' if pred == 1 else 'Other'

"""# Test"""

sample = {
    'Age': 30,
    'PurchaseAmount': 600,
    'Rating': 4,
    'Month': 6,
    'Gender': 'male'
}

print(predict_category(sample))