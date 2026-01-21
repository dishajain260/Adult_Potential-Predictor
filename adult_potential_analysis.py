#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv("adult.csv")


# In[3]:


df.head()
df.info()
df.describe()


# Handle missing values

# In[4]:


categorical_cols = df.select_dtypes(include=["object"]).columns
numerical_cols = df.select_dtypes(include=["number"]).columns


# In[5]:


from sklearn.impute import SimpleImputer

num_imp = SimpleImputer(strategy="mean")
df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])


# In[6]:


df['country'] = df['country'].astype(str).str.strip()
df['country'] = df['country'].replace('?', np.nan)


imp = SimpleImputer(strategy='most_frequent')
df[['country']] = imp.fit_transform(df[['country']])

df['country'].value_counts()


# In[7]:


# Clean text columns
df['education'] = df['education'].str.strip().str.title()
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

high_edu = ['Doctorate', 'Masters', 'Bachelors']
mid_edu  = ['Assoc-Acdm', 'Assoc-Voc', 'Some-College']

df['High_Potential'] = (
    df['education'].isin(high_edu).astype(int) * 2 +
    df['education'].isin(mid_edu).astype(int) +
    (df['hours-per-week'] >= 45).astype(int) +
    (df['capital-gain'] > 0).astype(int) +
    (df['country'] == 'United-States').astype(int) +
    (df['sex'] == 'Male').astype(int) +
    (df['race'] == 'White').astype(int)
    >= 5
).astype(int)
df['High_Potential']


# In[8]:


# Strip spaces and standardize capitalization
# Clean text columns
df['education'] = df['education'].str.strip().str.title()
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

high_edu = ['Doctorate', 'Masters', 'Bachelors']
mid_edu  = ['Assoc-Acdm', 'Assoc-Voc', 'Some-College']

df['High_Potential'] = (
    df['education'].isin(high_edu).astype(int) * 2 +
    df['education'].isin(mid_edu).astype(int) +
    (df['hours-per-week'] >= 45).astype(int) +
    (df['capital-gain'] > 0).astype(int) +
    (df['country'] == 'United-States').astype(int) +
    (df['sex'] == 'Male').astype(int) +
    (df['race'] == 'White').astype(int)
    >= 6
).astype(int)




# EDA :exploratory data analaysis

# In[9]:


# how balanced our classes are?

# df['High_Potential_Label'] = df['High_Potential'].map({1: 'HIGH', 0: 'LOW'})
# classes_count = df['High_Potential_Label'].value_counts()

# weighted_counts = df.groupby('High_Potential_Label')['fnlwgt'].sum()
# print(weighted_counts)

# plt.figure(figsize=(6,6))
# plt.pie(
#     weighted_counts,
#     labels=weighted_counts.index,
#     autopct="%1.1f%%",
#     startangle=90,
#     colors=["#ff9999", "#66b3ff"],
#     explode=(0.1, 0)  # highlight HIGH
# )
# plt.title("High Potential vs Low Potential (Population Weighted)")
# plt.axis('equal')
# plt.show()



# In[10]:


# gender_cnt = df["sex"].value_counts()
# ax = sns.barplot(gender_cnt)
# ax.bar_label(ax.containers[0])


# 

# In[11]:


# gender_cnt = df["race"].value_counts()
# ax = sns.barplot(gender_cnt)
# ax.bar_label(ax.containers[0])


# In[12]:


# gender_cnt = df["occupation"].value_counts()
# ax = sns.barplot(gender_cnt)
# ax.bar_label(ax.containers[0])


# In[13]:


# import seaborn as sns
# import matplotlib.pyplot as plt


# country_counts = df['country'].value_counts()


# plt.figure(figsize=(10,6))
# ax = sns.barplot(x=country_counts.index, y=country_counts.values)

# for p in ax.patches:
#     ax.annotate(
#         str(int(p.get_height())),  
#         (p.get_x() + p.get_width() / 2., p.get_height()),  
#         ha='center',
#         va='bottom',
#         fontsize=10
#     )

# plt.xticks(rotation=90) 
# plt.show()


# In[14]:


# fig, axes = plt.subplots(2, 2)

# sns.boxplot(ax=axes[0, 0], data=df, x="High_Potential",y="fnlwgt")
# # sns.boxplot(ax=axes[0, 1], data=df, x="High_Potential",y="country_score")
# # sns.boxplot(ax=axes[1, 0], data=df, x="High_Potential",y="capital_score")
# # sns.boxplot(ax=axes[1, 1], data=df, x="High_Potential",y="work_score")

# plt.tight_layout()


# In[15]:


# sns.histplot(
#     data=df,
#     x="education-num",
#     hue="High_Potential",
#     bins=20,
#     multiple="dodge"
# )


# In[16]:


# sns.histplot(
#     data=df,
#     x="age",
#     hue="High_Potential",
#     bins=20,
#     multiple="dodge"
# )


# encoding

# In[17]:


# df = df.drop("fnlwgt", axis=1)
# df = df.drop("High_Potential_1", axis=1)
df.head()
df.info()


# In[18]:

from sklearn.preprocessing import LabelEncoder

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le



# In[19]:

from sklearn.preprocessing import OneHotEncoder

cols = ['sex']

ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")

encoded = ohe.fit_transform(df[cols])

encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cols), index=df.index)

df = pd.concat([df.drop(columns=cols), encoded_df], axis=1)


# In[20]:


df.drop(columns=['potential_score_1'], errors='ignore', inplace=True)
df.drop(columns=['potential_score_2'], errors='ignore', inplace=True)
df.drop(columns=['potential_score_3'], errors='ignore', inplace=True)
df.drop(columns=['potential_score_4'], errors='ignore', inplace=True)
df.drop(columns=['potential_score_5'], errors='ignore', inplace=True)
df.head()
df.info()


# corelation heatmap

# In[21]:


num_cols = df.select_dtypes(include="number")
corr_matrix = num_cols.corr()

plt.figure(figsize=(15, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)


# training+feature scaling

# In[22]:


df.head()
df.info()
# df.drop(columns=['High_Potential_Label'], inplace=True)


# In[23]:


X = df.drop("High_Potential", axis=1)
y = df["High_Potential"]


# In[24]:


y.head()


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[26]:


X_test.head()


# In[27]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[28]:


X_test_scaled


# Train and Evaluate models

# In[29]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)

y_pred = log_model.predict(X_test_scaled)

# Evaluation
print("Logistic Regression Model")
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 score: ", f1_score(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("CM: ", confusion_matrix(y_test, y_pred))


# In[30]:


from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_scaled, y_train)

y_pred = knn_model.predict(X_test_scaled)

# Evaluation
print("KNN Model")
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 score: ", f1_score(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("CM: ", confusion_matrix(y_test, y_pred))


# In[31]:


# Naive Bayes

from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

y_pred = nb_model.predict(X_test_scaled)

# Evaluation
print("Naive Bayes Model")
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 score: ", f1_score(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("CM: ", confusion_matrix(y_test, y_pred))


# Feature Engineering

# In[32]:


# Add or Tranform features
df["education-num_sq"] = df["education-num"] ** 2
df["capital-gain_sq"] = df["capital-gain"] ** 2
df["salary_sq"] = df["salary"] ** 2

# df["Applicant_Income_log"] = np.log1p(df["Applicant_Income"])

X = df.drop(columns=["High_Potential", "education-num", "capital-gain","salary"])
y = df["High_Potential"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[33]:


X_train.head()


# In[34]:


# Logistic regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)

y_pred = log_model.predict(X_test_scaled)

# Evaluation
print("Logistic Regression Model")
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 score: ", f1_score(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("CM: ", confusion_matrix(y_test, y_pred))


# In[35]:


# KNN

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_scaled, y_train)

y_pred = knn_model.predict(X_test_scaled)

# Evaluation
print("KNN Model")
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 score: ", f1_score(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("CM: ", confusion_matrix(y_test, y_pred))


# In[36]:


# Naive Bayes

from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

y_pred = nb_model.predict(X_test_scaled)

# Evaluation
print("Naive Bayes Model")
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 score: ", f1_score(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("CM: ", confusion_matrix(y_test, y_pred))


# In[ ]:

import pickle

pickle.dump(knn_model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(label_encoders, open("label_encoders.pkl", "wb"))
pickle.dump(ohe, open("ohe.pkl", "wb"))




