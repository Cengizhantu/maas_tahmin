# İş Problemi:
# Maaş bilgileri ve 1986 yılına ait kariyer istatistikleri paylaşılan beyzbol
# oyuncularının maaş tahminleri için bir makine öğrenmesi modeli geliştiriniz.

# Veri seti hikayesi:
# Bu veri seti orijinal olarak Carnegie Mellon Üniversitesi'nde bulunan
# StatLib kütüphanesinden alınmıştır. Veri seti 1988 ASA Grafik Bölümü
# Poster Oturumu'nda kullanılan verilerin bir parçasıdır. Maaş verileri
# orijinal olarak Sports Illustrated, 20 Nisan 1987'den alınmıştır. 1986 ve
# kariyer istatistikleri, Collier Books, Macmillan Publishing Company,
# New York tarafından yayınlanan 1987 Beyzbol Ansiklopedisi
# Güncellemesinden elde edilmiştir.
""" Değişkenler"""
# AtBat: 1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş sayısı
# Hits: 1986-1987 sezonundaki isabet sayısı
# HmRun: 1986-1987 sezonundaki en değerli vuruş sayısı
# Runs: 1986-1987 sezonunda takımına kazandırdığı sayı
# RBI: Bir vurucunun vuruş yaptıgında koşu yaptırdığı oyuncu sayısı
# Walks: Karşı oyuncuya yaptırılan hata sayısı
# Years: Oyuncunun major liginde oynama süresi (sene)
# CAtBat: Oyuncunun kariyeri boyunca topa vurma sayısı
# CHits: Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı
# CHmRun: Oyucunun kariyeri boyunca yaptığı en değerli sayısı
# CRuns: Oyuncunun kariyeri boyunca takımına kazandırdığı sayı
# CRBI: Oyuncunun kariyeri boyunca koşu yaptırdırdığı oyuncu sayısı
# CWalks: Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı
# League: Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N seviyelerine sahip bir faktör
# Division: 1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve W seviyelerine sahip bir faktör
# PutOuts: Oyun icinde takım arkadaşınla yardımlaşma
# Assits:1986-1987 sezonunda oyuncunun yaptığı asist sayısı
# Errors: 1986-1987 sezonundaki oyuncunun hata sayısı
# Salary: Oyuncunun 1986-1987 sezonunda aldığı maaş(bin uzerinden)
# NewLeague: 1987 sezonunun başında oyuncunun ligini gösteren A ve N seviyelerine sahip bir faktör

# Görev: Veri ön işleme ve özellik mühendisliği gerçekleştirerek makine öğrenmesi modelini oluşturunuz.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import RobustScaler
import matplotlib
import sklearn
matplotlib.use('Qt5Agg')
pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_csv("../datasets/hitters.csv")


##################################
# GÖREV 1: KEŞİFCİ VERİ ANALİZİ
##################################

##################################
# GENEL RESİM
##################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)


##################################
# NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
##################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.
    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri
    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi
    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))
    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


##################################
# KATEGORİK DEĞİŞKENLERİN ANALİZİ
##################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col, plot=True)


##################################
# NUMERİK DEĞİŞKENLERİN ANALİZİ
##################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, plot=True)


##################################
# NUMERİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
##################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, "Salary", col)


##################################
# KATEGORİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
##################################

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "TARGET_COUNT": dataframe.groupby(categorical_col)[target].count()}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df, "Salary", col)

##################################
# KORELASYON
##################################
df.corr()


def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    corr_matrix = corr.abs()
    upper_triangle_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > 0.90)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={"figure.figsize": (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list


high_correlated_cols(df, plot=True)

drop_list = high_correlated_cols(df)

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, )
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

#################################
# GÖREV 2: FEATURE ENGINEERING
##################################

##################################
# EKSİK DEĞER ANALİZİ
##################################
df.isnull().sum()


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


na_columns = missing_values_table(df, na_name=True)

# df["SALARY"].fillna(df["SALARY"].median(), inplace=True)

imputer = KNNImputer(n_neighbors=5)

df[num_cols] = pd.DataFrame(imputer.fit_transform(df[num_cols]), columns=num_cols)
df.head()

na_columns = missing_values_table(df, na_name=True)


##################################
# AYKIRI DEĞER ANALİZİ
##################################

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
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


def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    print(col, check_outlier(df, col))

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df[num_cols])
df_scores = clf.negative_outlier_factor_
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style=".-")
plt.show()
np.sort(df_scores)[0:10]
th = np.sort(df_scores)[2]
df[df_scores < th].shape

##################################
# ÖZELLİK ÇIKARIMI
#################################

# Career RATES
df["NEW_C_HIT_RATIO"] = df["Hits"] / df["CHits"]
df["NEW_C_ATBAT_RATIO"] = df["AtBat"] / df["CAtBat"]
df["NEW_C_HMRUN_RATIO"] = df["HmRun"] / df["CHmRun"]
df["NEW_C_RUN_RATIO"] = df["Runs"] / df["CRuns"]
df["NEW_C_WALK_RATIO"] = df["Walks"] / df["CWalks"]
df["NEW_C_RBI_RATIO"] = df["RBI"] / df["CRBI"]

# LEAGUE VS. NEW LEAGUE
df.loc[(df.League == "A") & (df.NewLeague == "A"), "NEW_PLAYER_PROGRESS"] = "StandA"
df.loc[(df.League == "N") & (df.NewLeague == "N"), "NEW_PLAYER_PROGRESS"] = "StandN"
df.loc[(df.League == "A") & (df.NewLeague == "N"), "NEW_PLAYER_PROGRESS"] = "A_N"
df.loc[(df.League == "N") & (df.NewLeague == "A"), "NEW_PLAYER_PROGRESS"] = "N_A"
df.groupby("NEW_PLAYER_PROGRESS").agg({"Salary": "mean"})
df.columns

# NEW_YEARS

df.loc[df["Years"] <= 2, "NEW_YEARS"] = "junior_player"
df.loc[(df["Years"] > 2) & (df["Years"] <= 5), "NEW_YEARS"] = "mid_player"
df.loc[(df["Years"] > 5) & (df["Years"] <= 10), "NEW_YEARS"] = "senior_player"
df.loc[df["Years"] > 10, "NEW_YEARS"] = "expert_player"
df.groupby("NEW_YEARS").agg({"Salary": "mean"}).sort_values("Salary", ascending=False)

# AVERAGE
df["NEW_AVG_C_HITS"] = df["CHits"] / df["Years"]
df["NEW_AVG_C_ATBAT"] = df["CAtBat"] / df["Years"]
df["NEW_AVG_C_HMRUN"] = df["CHmRun"] / df["Years"]
df["NEW_AVG_C_RUNS"] = df["CRuns"] / df["Years"]
df["NEW_AVG_C_WALKS"] = df["CWalks"] / df["Years"]
df["NEW_AVG_C_RBI"] = df["CRBI"] / df["Years"]

# PLAYER_PROGRESS X DIVISION
df.loc[(df["NEW_YEARS"] == "junior_player") & (df["Division"] == "E"), "NEW_DIVISION_CAT"] = "junior_east"
df.loc[(df["NEW_YEARS"] == "junior_player") & (df["Division"] == "W"), "NEW_DIVISION_CAT"] = "junior_west"
df.loc[(df["NEW_YEARS"] == "mid_player") & (df["Division"] == "E"), "NEW_DIVISION_CAT"] = "mid_east"
df.loc[(df["NEW_YEARS"] == "mid_player") & (df["Division"] == "W"), "NEW_DIVISION_CAT"] = "mid_west"
df.loc[(df["NEW_YEARS"] == "senior_player") & (df["Division"] == "E"), "NEW_DIVISION_CAT"] = "senior_east"
df.loc[(df["NEW_YEARS"] == "senior_player") & (df["Division"] == "W"), "NEW_DIVISION_CAT"] = "senior_west"
df.loc[(df["NEW_YEARS"] == "expert_player") & (df["Division"] == "E"), "NEW_DIVISION_CAT"] = "expert_east"
df.loc[(df["NEW_YEARS"] == "expert_player") & (df["Division"] == "W"), "NEW_DIVISION_CAT"] = "expert_west"

df.groupby("NEW_DIVISION_CAT").agg({"Salary": ["mean", "count"]})

# Oyuncunun kariyeri boyunca yaptığı başarılı işlerin oynadığı süreye göre ortalaması
df['NEW_CAREER_YEARS'] = (df['CAtBat'] + df['CHits'] + df['CHmRun'] + df['CRuns'] + df['CRBI'] + df['CWalks']) / df[
    'Years']

cat_cols, num_cols, cat_but_car = grab_col_names(df)
for col in num_cols:
    print(col, check_outlier(df, col))


# Encoding
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"] and df[col].nunique() == 2]
for col in binary_cols:
    df = label_encoder(df, col)


##############################
# 5. Rare Encoding
##############################
grab_col_names(df)
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))  # Kategorik değişkenler ve bunların eşsiz değer sayısı.
        print(pd.DataFrame(
            {"COUNT": dataframe[col].value_counts(), "RATIO": dataframe[col].value_counts() / len(dataframe),
             "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


rare_analyser(df, "Salary", cat_cols)


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, cat_cols, drop_first=True)

# Ölçeklendirme

##########################################################
# 7. Min Max Scaler
#########################################################
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df.head()

for col in num_cols:
    if check_outlier(df, col):
        replace_with_thresholds(df, col )

for col in num_cols:
    print(col, check_outlier(df, col))

for col in df.columns:
    print(col, check_outlier(df, col))
df_null1 = df[df["NEW_C_HMRUN_RATIO"].isnull()]
df_null2 = df[df["NEW_C_WALK_RATIO"].isnull()]
df_null2 = df[df["NEW_C_RBI_RATIO"].isnull()]
df.dropna(inplace=True)

missing_values_table(df)
# Model
#########################
X= df.drop(["Salary"], axis=1)
y= df[["Salary"]].values.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

reg_model=LinearRegression()
reg_model.fit(X_train, y_train)

y_pred=reg_model.predict(X_train)
np.sqrt((mean_squared_error(y_train,y_pred)))

# 231.461
# TRAIN RKARE: Bağımsız değişkenlerin bağımlı değerleri etkileme,açıklama oranı

reg_model.score(X_train,y_train)
#0.7156

# Test RMSE
y_pred=reg_model.predict(X_train)
np.sqrt((mean_squared_error(y_train,y_pred)))

#231.461

# Test RKARE
reg_model.score(X_test,y_test)

# 0.5141

# 10 katlı CV (CROSS VALIDATION)
np.mean(np.sqrt(-cross_val_score(reg_model,X,y,cv=10,scoring="neg_mean_absolute_error")))
# 14.19

# 10 katlı CV sonucunda modelin maaşları +-14.19 ile doğru tahmin edebildiğini gözlemledik.

##########################
# BASE MODELS
##########################

def all_models(X, y, test_size=0.2, random_state=12345, classification=True):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
        roc_auc_score, confusion_matrix, classification_report, plot_roc_curve, mean_squared_error

    # Tum Base Modeller (Classification)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.svm import SVC

    # Tum Base Modeller (Regression)
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state)
    all_models = []

    if classification:
        models = [('LR', LogisticRegression(random_state=random_state)),
                  ('KNN', KNeighborsClassifier()),
                  ('CART', DecisionTreeClassifier(random_state=random_state)),
                  ('RF', RandomForestClassifier(random_state=random_state)),
                  ('SVM', SVC(gamma='auto', random_state=random_state)),
                  ('XGB', GradientBoostingClassifier(random_state=random_state)),
                  ("LightGBM", LGBMClassifier(random_state=random_state)),
                  ("CatBoost", CatBoostClassifier(verbose=False, random_state=random_state))]

        for name, model in models:
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            acc_train = accuracy_score(y_train, y_pred_train)
            acc_test = accuracy_score(y_test, y_pred_test)
            values = dict(name=name, acc_train=acc_train, acc_test=acc_test)
            all_models.append(values)

        sort_method = False
    else:
        models = [('LR', LinearRegression()),
                  ("Ridge", Ridge()),
                  ("Lasso", Lasso()),
                  ("ElasticNet", ElasticNet()),
                  ('KNN', KNeighborsRegressor()),
                  ('CART', DecisionTreeRegressor()),
                  ('RF', RandomForestRegressor()),
                  ('SVR', SVR()),
                  ('GBM', GradientBoostingRegressor()),
                  ("XGBoost", XGBRegressor()),
                  ("LightGBM", LGBMRegressor()),
                  ("CatBoost", CatBoostRegressor(verbose=False))]

        for name, model in models:
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            values = dict(name=name, RMSE_TRAIN=rmse_train, RMSE_TEST=rmse_test)
            all_models.append(values)

        sort_method = True
    all_models_df = pd.DataFrame(all_models)
    all_models_df = all_models_df.sort_values(all_models_df.columns[2], ascending=sort_method)
    print(all_models_df)
    return all_models_df

all_models = all_models(X, y, test_size=0.2, random_state=46, classification=False)


#Maaş'a bağlı korelasyon

corr_matrix = df.corr()
corr_matrix["Salary"].sort_values(ascending=False)

##########################
# RANDOM FORESTS MODEL TUNING
##########################

# Tuning için hazırlanan parametreler. Tuning zaman aldığı için çıkan parametre değerlerini girdim.
rf_params = {"max_depth": [4, 5, 7, 10],
             "max_features": [4, 5, 6, 8, 10, 12],
             "n_estimators": [80, 100, 150, 250, 400, 500],
             "min_samples_split": [8, 10, 12, 15]}

# rf_cv_model = GridSearchCV(rf_model, rf_params, cv = 10, n_jobs = -1).fit(X_train , y_train)
# rf_cv_model.best_params_

best_params = {'max_depth': 10,
               'max_features': 8,
               'min_samples_split': 10,
               'n_estimators': 80}

rf_model = RandomForestRegressor(random_state=42).fit(X_train, y_train)

# RANDOM FORESTS TUNED MODEL
rf_tuned = RandomForestRegressor(max_depth=10, max_features=4, n_estimators=150,
                                 min_samples_split=8, random_state=42).fit(X_train, y_train)

# TUNED MODEL TRAIN HATASI
y_pred = rf_tuned.predict(X_train)

print("RF Tuned Model Train RMSE:", np.sqrt(mean_squared_error(y_train, y_pred)))

##########################
# TUNED MODEL TEST HATASI
##########################

y_pred = rf_tuned.predict(X_test)
print("RF Tuned Model Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

def plot_importance(model, features, num=len(X), save=False):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_tuned, X_train)


# Tuned edilmiş model nesnesinin kaydedilmesi
import pickle
pickle.dump(rf_tuned, open("rf_final_model.pkl", 'wb'))

# Tuned edilmiş model nesnesinin yüklenmesi
df_prep = pickle.load(open('rf_final_model.pkl', 'rb'))


