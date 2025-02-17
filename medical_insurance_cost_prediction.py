############################################
# 1. Gerekli Kütüphaneleri Yükleme
############################################
# Temel Kütüphaneler
import numpy as np
import pandas as pd
import time
import warnings
import os
import pickle
import streamlit as st

# Veri Görselleştirme
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()
# Scikit-learn preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

# Scikit-learn metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Temel Modeller
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Boosting Modelleri
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

##########################################
# 2. Veri Ön İşleme Ayarları
##########################################
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Görselleştirme ayarları
sns.set_theme()  # Seaborn temasını ayarla
plt.rcParams['figure.figsize'] = (10, 6)

# Warning mesajlarını kapatma
warnings.filterwarnings('ignore')

#########################################
# 3. Veri Setini Yükleme
#########################################

# CSV dosyasını okuma
df = pd.read_csv(r"C:\Users\ASUS\Desktop\Medical_Insurance_Cost_Prediction\insurance.csv")

##############################
# Veriye ilk bakış
##############################

def check_df(dataframe, head=5):
    print('##################### Shape #####################')
    print(dataframe.shape)
    print('##################### Types #####################')
    print(dataframe.dtypes)
    print('##################### Head #####################')
    print(dataframe.head(head))
    print('##################### Tail #####################')
    print(dataframe.tail(head))
    print('##################### NA #####################')
    print(dataframe.isnull().sum())
    print('##################### Quantiles #####################')
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

#################
# Kategorik ve Nümerik Değişkenlerin Tespiti
#################

def grab_col_names(dataframe, cat_th=5, car_th=100):
    """

    Returns the names of categorical, numeric and categorical but cardinal variables in the data set.
    Note Categorical variables include categorical variables with numeric appearance.

    Parameters
    ------
        dataframe: dataframe
                Variable names of the dataframe to be taken
        cat_th: int, optional
                class threshold for numeric but categorical variables
        car_th: int, optinal
                class threshold for categorical but cardinal variables

    Returns
    ------
        cat_cols: list
                Categorical variable list
        num_cols: list
                Numeric variable list
        cat_but_car: list
                List of cardinal variables with categorical appearance

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is inside cat_cols.
        The sum of the 3 return lists equals the total number of variables: cat_cols + num_cols + cat_but_car = number of variables

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car, num_but_cat

cat_cols, num_cols, cat_but_car,  num_but_cat = grab_col_names(df)

df.head(5)
df['children'].nunique()
# 6

cat_cols
num_cols
cat_but_car
num_but_cat

###########################
# Kategorik Değişken Analizi
##############################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        'Ratio': 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print('##########################################')
    if plot:
        plt.figure(figsize=(12,6))
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col, plot=True)

#######################
# Nümerik Değişken Analizi
#######################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=False)

###################
# Hedef Değişken Analizi
###################

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({'TARGET_MEAN': dataframe.groupby(categorical_col)[target].mean()}), end='\n\n\n')

for col in cat_cols:
    target_summary_with_cat(df, 'charges', col)

################
# Korelasyon Analizi
#################

def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    # Sadece numerik kolonları seç
    num_cols = [col for col in dataframe.columns if dataframe[col].dtype in ['int64', 'float64']]
    corr = dataframe[num_cols].corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]

    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="RdBu", annot=True, fmt=".2f")
        plt.title('Correlation Matrix of Numeric Variables')
        plt.show()

    return drop_list

# Korelasyon analizi
print("Yüksek Korelasyonlu Değişkenler:")
high_correlated_cols(df, plot=True)

######################################
# Hedef Değişken (Charges) Dağılımı
######################################

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
df["charges"].hist(bins=50)
plt.title("Charges Distribution")

plt.subplot(1, 2, 2)
np.log1p(df['charges']).hist(bins=50)
plt.title("Log Transformed Charges Distribution")
plt.show(block=True)

# Ayrıca smoker'a göre charges dağılımına bakalım
plt.figure(figsize=(12, 6))
sns.boxplot(x='smoker', y='charges', data=df)
plt.title('Charges Distribution by Smoking Status')
plt.show(block=True)

# BMI ve age'e göre charges dağılımı
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(df['age'], df['charges'], alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Charges')
plt.title('Age vs Charges')

plt.subplot(1, 2, 2)
plt.scatter(df['bmi'], df['charges'], alpha=0.5)
plt.xlabel('BMI')
plt.ylabel('Charges')
plt.title('BMI vs Charges')
plt.tight_layout()
plt.show(block=True)

df['charges'].hist(bins=100)
plt.show(block=True)

###################
# Outliers Analysis
####################

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

check_outlier(df, num_cols)

#######################
# Missing Value Analysis
#######################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)

    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)

##########################
# Rare Analysis
#########################

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                           "RATIO": dataframe[col].value_counts() / len(dataframe),
                           "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "charges", cat_cols)

"""
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])
    return temp_df

rare_encoder(df, 0.01)

# bunları yapmadık çünkü veri seti dengeli

"""

###########################
# Feature Extraction
###########################

###########################
# Feature Extraction
###########################

# 1. BMI Kategorileri (WHO standartlarına göre)
df['bmi_cat'] = pd.cut(df['bmi'],
                      bins=[0, 18.5, 24.9, 29.9, float('inf')],
                      labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

# 2. Yaş Grupları
df['age_cat'] = pd.cut(df['age'],
                      bins=[0, 30, 45, 65],
                      labels=['Young', 'Middle', 'Senior'])

# 3. Risk Skoru (Sigara ve BMI kombinasyonu)
df['risk_score'] = df.apply(lambda x: 3 if (x['smoker'] == 'yes' and x['bmi'] > 30)
                           else 2 if (x['smoker'] == 'yes' and x['bmi'] > 25)
                           else 1 if (x['smoker'] == 'yes' or x['bmi'] > 30)
                           else 0, axis=1)

# 4. Aile Durumu
df['family_size'] = df['children'] + 1  # kişinin kendisi
df['has_children'] = (df['children'] > 0).astype(int)

# 5. BMI ve Yaş Etkileşimi
df['age_bmi_interaction'] = (df['age'] * df['bmi']) / 100

# Yeni değişkenleri kontrol edelim
print("\nYeni değişkenlerin ilk birkaç satırı:")
new_features = ['bmi_cat', 'age_cat', 'risk_score', 'family_size', 'has_children',
                'age_bmi_interaction']
print(df[new_features].head())

print("\nYeni değişkenlerin özet istatistikleri:")
print(df[new_features].describe())

######################
# Encoding
#####################
cat_cols, num_cols, cat_but_car,  num_but_cat = grab_col_names(df)
df.head()

cat_cols
num_cols
cat_but_car
num_but_cat

# Kategorik değişkenlerimiz
cat_cols = ['sex', 'smoker', 'region', 'bmi_cat', 'age_cat', 'risk_score', 'has_children']

# Numerik değişkenlerimiz
num_cols = ['age', 'bmi', 'children', 'charges', 'family_size', 'age_bmi_interaction']

cat_but_car = []
num_but_cat = []
df.shape

# One-Hot Encoding
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()
df.shape
####################
# Standardization Process
####################

num_cols = [col for col in num_cols if col not in ["charges"]]

scaler = RobustScaler()

df[num_cols] = scaler.fit_transform(df[num_cols])

df.head(10)

######################
# Creating Model
######################

# Bağımlı ve bağımsız değişkenleri ayıralım
y = df["charges"]
X = df.drop(["charges"], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

# LightGBM için base parametreler
lgbm_params = {
    'verbose': -1,
    'force_row_wise': True,
    'n_jobs': -1
}

# Modellerimizi tanımlayalım
models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('GBM', GradientBoostingRegressor()),
          ("LightGBM", LGBMRegressor(**lgbm_params)),
          ("CatBoost", CatBoostRegressor(verbose=False))]

# Sonuçları saklamak için listeler
rmse_scores = []
r2_scores = []
mae_scores = []
mse_scores = []
execution_times = []

# Modelleri eğit ve değerlendir
for name, regressor in models:
    start_time = time.time()

    # Model eğitimi
    regressor.fit(X_train, y_train)

    # Tahminler
    y_pred = regressor.predict(X_test)

    # RMSE hesaplama
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5,
                                            scoring="neg_mean_squared_error")))
    rmse_scores.append(rmse)

    # R2 score hesaplama
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)

    # MAE hesaplama
    mae = mean_absolute_error(y_test, y_pred)
    mae_scores.append(mae)

    # MSE hesaplama
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)

    # Çalışma süresini hesaplama
    execution_time = time.time() - start_time
    execution_times.append(execution_time)

    print(f"Model: {name}")
    print(f"RMSE: {round(rmse, 4)}")
    print(f"R^2 Score: {round(r2, 4)}")
    print(f"MAE: {round(mae, 4)}")
    print(f"MSE: {round(mse, 4)}")
    print(f"Execution Time: {round(execution_time, 2)} seconds\n")

# En iyi modeli gösterme
best_model_idx = np.argmax(r2_scores)
print("\n" + "=" * 50)
print("\033[1mEn İyi Model:")
print(f"Model: {models[best_model_idx][0]}")
print(f"RMSE: {round(rmse_scores[best_model_idx], 4)}")
print(f"R^2 Score: {round(r2_scores[best_model_idx], 4)}")
print(f"MAE: {round(mae_scores[best_model_idx], 4)}")
print(f"MSE: {round(mse_scores[best_model_idx], 4)}")
print(f"Execution Time: {round(execution_times[best_model_idx], 2)} seconds\033[0m")

####################
# Hyperparameter Optimization
####################

# Lasso için parametre grid'i
lasso_params = {
    'alpha': [0.001, 0.005, 0.01, 0.05, 0.1]
}

# GBM için parametre grid'i
gbm_params = {
    'learning_rate': [0.03, 0.05, 0.07],
    'n_estimators': [150, 200, 250],
    'max_depth': [3, 4],
    'subsample': [0.8, 0.9],
    'min_samples_split': [3, 5]
}

# Modelleri ve parametreleri bir sözlükte tutalım
models = {
    'Lasso': (Lasso(), lasso_params),
    'GBM': (GradientBoostingRegressor(), gbm_params)
}

# Sonuçları saklamak için listeler
best_models = {}
best_scores = {}
execution_times = {}

for name, (model, params) in models.items():
    print(f"\nHyperparameter Tuning for {name}:")
    print("-" * 50)

    start_time = time.time()

    # Grid Search
    grid_search = GridSearchCV(model,
                               param_grid=params,
                               cv=5,
                               n_jobs=-1,
                               scoring='neg_mean_squared_error')

    grid_search.fit(X_train, y_train)

    # En iyi modeli sakla
    best_models[name] = grid_search.best_estimator_

    # Tahminler
    y_pred = grid_search.predict(X_test)

    # Metrikleri hesapla
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    execution_time = time.time() - start_time
    execution_times[name] = execution_time

    # Sonuçları yazdır
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score (MSE): {-grid_search.best_score_:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R^2 Score: {r2:.4f}")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"Execution Time: {execution_time:.2f} seconds")

# En iyi modeli bul
best_model_name = max(best_models.keys(), key=lambda x: r2_score(y_test, best_models[x].predict(X_test)))

print("\n" + "=" * 50)
print("\033[1mEn İyi Model:")
print(f"Model: {best_model_name}")
y_pred_best = best_models[best_model_name].predict(X_test)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_best)):.2f}")
print(f"R^2 Score: {r2_score(y_test, y_pred_best):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_best):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_best):.2f}")
print(f"Execution Time: {execution_times[best_model_name]:.2f} seconds\033[0m")

#####################################
# Final Model ve Tahmin
#####################################

# En iyi GBM parametreleri ile final modeli oluşturalım
final_model = GradientBoostingRegressor(
   learning_rate=0.03,
   max_depth=3,
   min_samples_split=5,
   n_estimators=150,
   subsample=0.9,
   random_state=17
)

# Modeli eğitelim
final_model.fit(X_train, y_train)

# Test seti üzerinde tahmin
y_pred = final_model.predict(X_test)

# Sonuçları DataFrame'e dönüştürme
results = pd.DataFrame({
  'True Price': y_test,
  'Predicted Price': y_pred,
  'Difference': y_test - y_pred,
  'Absolute Difference': abs(y_test - y_pred),
  'Percentage Error': abs((y_test - y_pred) / y_test) * 100
})

# Özet istatistikler
print("\nModel Performance Metrics:")
print("-" * 50)
print(f"Mean Absolute Error: ${results['Absolute Difference'].mean():,.2f}")
print(f"Mean Percentage Error: %{results['Percentage Error'].mean():.2f}")
print(f"Median Absolute Error: ${results['Absolute Difference'].median():,.2f}")
print(f"Median Percentage Error: %{results['Percentage Error'].median():.2f}")

# En büyük 5 hata
print("\nTop 5 Largest Prediction Errors:")
print("-" * 50)
print(results.nlargest(5, 'Absolute Difference'))

# En küçük 5 hata
print("\nTop 5 Most Accurate Predictions:")
print("-" * 50)
print(results.nsmallest(5, 'Absolute Difference'))

########################
# Modeli Kaydetme
########################

# Proje dizinini belirle
project_dir = r"C:\Users\ASUS\Desktop\Medical_Insurance_Cost_Prediction"

# Models klasörünü oluştur
models_dir = os.path.join(project_dir, 'models')
os.makedirs(models_dir, exist_ok=True)

# Model yolunu belirle
model_path = os.path.join(models_dir, 'final_model.pkl')

# Modeli kaydet
with open(model_path, 'wb') as file:
  pickle.dump(final_model, file)

print(f"\nModel başarıyla kaydedildi: {model_path}")

# Feature'ları kaydetme
feature_columns = list(X.columns)
feature_columns_path = os.path.join(models_dir, "feature_columns.pkl")

with open(feature_columns_path, "wb") as file:
   pickle.dump(feature_columns, file)

print(f"Feature columns başarıyla kaydedildi: {feature_columns_path}")

# Scaler'ı kaydet
scaler_path = os.path.join(models_dir, "scaler.pkl")

with open(scaler_path, "wb") as file:
   pickle.dump(scaler, file)

print(f"Scaler başarıyla kaydedildi: {scaler_path}")

















