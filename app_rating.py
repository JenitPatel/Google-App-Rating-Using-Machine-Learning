# Importing all necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load Dataset
df = pd.read_csv('C:\\Users\\Jenit\\Desktop\\Google Apps\\googleplaystore.csv')

# Overlook datatset with all attributes and its value
print(df.head())

# Get a concise summary of the dataframe
print(df.info())

# To check any null values in any of the attributes in the dataset
print(df.isnull().sum())

# Drop rows with null values
df.dropna(inplace = True)

# Drop columns 'Current Ver','Android Ver','App' from the dataset
df.drop(labels = ['Current Ver','Android Ver','App'], axis = 1, inplace = True)

# Overlook datatset with all attributes and its value
print(df.head())

# Converting categorical variables to numeric so that they can work smoothly with machine learning algorithms.
category_list = df['Category'].unique().tolist()
category_list = ['cat_' + word for word in category_list]
df = pd.concat([df, pd.get_dummies(df['Category'], prefix='cat')], axis=1)

# Overlook datatset with all attributes and its value
print(df.head())

# Label Encoding for all necessary attributes
le = preprocessing.LabelEncoder()
df['Genres'] = le.fit_transform(df['Genres'])

le = preprocessing.LabelEncoder()
df['Content Rating'] = le.fit_transform(df['Content Rating'])


df['Price'] = df['Price'].apply(lambda x : x.strip('$'))
df['Installs'] = df['Installs'].apply(lambda x : x.strip('+').replace(',', ''))
df['Type'] = pd.get_dummies(df['Type'])
df["Size"] = [str(round(float(i.replace("k", ""))/1024, 3)) if "k" in i else i for i in df.Size]
df['Size'] = df['Size'].apply(lambda x: x.strip('M'))
df[df['Size'] == 'Varies with device'] = 0
df['Size'] = df['Size'].astype(float)
df["Size"]

df['new'] = pd.to_datetime(df['Last Updated'])
df['lastupdate'] = (df['new'] -  df['new'].max()).dt.days

# Data Preparation for modelling
x = df.drop(labels=["Rating","Category", "Last Updated", "new"], axis = 1)
y = df['Rating']

# Overlook all features that will be feed to modelling
print(x.head())


# Splitting train and test dataset for modelling
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 42)

# Algorithm 1 : Linear Regression
multiple_linear_regression = LinearRegression()

# Training model
multiple_linear_regression.fit(x_train, y_train)

# Performance computing
accuracy = multiple_linear_regression.score(x_test,y_test)
mlr_accuracy = float(str(np.round(accuracy*100, 2)))
mlr_error = np.round(100 - mlr_accuracy, 2)
print('Accuracy of Linear Regression : ' + str(np.round(accuracy*100, 2)) + '%')



# Algorithm 2 : K-Nearest Neighbor Algorithm
knn = KNeighborsRegressor(n_neighbors=25)

# Training model
knn.fit(x_train, y_train)

# Performance computing
accuracy = knn.score(x_test,y_test)
knn_accuracy = float(str(np.round(accuracy*100, 2)))
knn_error = np.round(100 - knn_accuracy, 2)
print('Accuracy of K-Nearest Neighbor : ' + str(np.round(accuracy*100, 2)) + '%')

# Plotting to determine 'K' neighbors
n_neighbors = np.arange(1, 40, 1)
scores = []
for n in n_neighbors:
    knn.set_params(n_neighbors=n)
    knn.fit(x_train, y_train)
    scores.append(knn.score(x_test, y_test))
plt.figure(figsize=(7, 5))
plt.title("Effect of Estimators")
plt.xlabel("Number of Neighbors K")
plt.ylabel("Score")
plt.plot(n_neighbors, scores)
plt.show()

print("K value to achieve this result : ", n_neighbors[scores.index(max(scores))])



# Algorithm 3 : Random Forest Algorithm
rfr = RandomForestRegressor()

# Training model
rfr.fit(x_train, y_train)

# Performance computing
accuracy = rfr.score(x_test, y_test)
rf_accuracy = float(str(np.round(accuracy*100, 2)))
rf_error = np.round(100 - rf_accuracy, 2)
print('Accuracy of Random Forest : ' + str(np.round(accuracy*100, 2)) + '%')

# Plotting to determine number of estimators
rf = RandomForestRegressor(n_jobs=-1)
estimators = np.arange(10, 150, 10)
scores = []
for n in estimators:
    rf.set_params(n_estimators=n)
    rf.fit(x_train, y_train)
    scores.append(rf.score(x_test, y_test))
plt.figure(figsize=(7, 5))
plt.title("Effect of Estimators")
plt.xlabel("no. estimator")
plt.ylabel("score")
plt.plot(estimators, scores)
plt.show()

print("the number of estimators required to achieve this result: ", estimators[scores.index(max(scores))])

# Comparing all models
import matplotlib.ticker as mticker

labels = ['Linear Regression', 'K-Nearest Neighbor', 'Random Forest']

accuracy = [mlr_accuracy, knn_accuracy, rf_accuracy]
error = [mlr_error, knn_error, rf_error]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, accuracy, width, label='Accuracy')
rects2 = ax.bar(x + width/2, error, width, label='Misclassification Error')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Machine Learning Algorithms')
ax.set_ylabel('Accuracy and Misclassification Error (%)')
ax.set_title('Model Comparison')
ax.set_xticks(x)
ax.set_ylim(0,110)
ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=12, prune='upper'))
ax.set_xticklabels(labels,fontsize=8.8)
ax.legend(loc="upper left",bbox_to_anchor=(1,1))

# Attach a text label above each bar in *rects*, displaying its height
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.show()
