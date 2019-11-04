import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def heat_map(data, row, col, output, func):
    heat_map_data = pd.pivot_table(data,
                                   values=output, index=row, columns=col,
                                   aggfunc=func)
    print(heat_map_data)
    print('-'*20)
    sns.heatmap(heat_map_data, cmap="RdBu")
    # plt.show()

def clean_data(data):

    # fill missing values
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

    # create bands for grouping continuous values
    data['AgeBand'] = pd.cut(data['Age'].astype(int), 5)
    data['FareBand'] = pd.qcut(data['Fare'], 5)

    # feature engineering
    data['FamSize'] = data['SibSp'] + data['Parch'] + 1
    data['Title'] = \
        data['Name'].str.split(',', expand=True)[1].str.split('.', expand=True)[0]

    title_rare_or_not = data['Title'].value_counts() < 10
    data['Title'] = data['Title'].apply(lambda x: 'Misc'
        if title_rare_or_not.loc[x] else x)

    # quantify categorical features
    from sklearn.preprocessing import LabelEncoder
    label = LabelEncoder()

    data['Sex'] = label.fit_transform(data['Sex'])
    data['Embarked'] = label.fit_transform(data['Embarked'])
    data['AgeBand'] = label.fit_transform(data['AgeBand'])
    data['FareBand'] = label.fit_transform(data['FareBand'])
    data['Title'] = label.fit_transform(data['Title'])

    # drop unusable feature(s)
    data.drop(['Cabin', 'Ticket', 'Name', 'Fare',
               'Age', 'SibSp', 'Parch', 'PassengerId'],
              axis=1, inplace=True)

    return data


def get_data():
    raw_data = pd.read_csv('~/Desktop/ML-Data/titanic/train.csv')
    test_data = pd.read_csv('~/Desktop/ML-Data/titanic/test.csv')
    return [raw_data, test_data]


def train_model(data):
    X = data.drop('Survived', axis=1)
    y = data['Survived']

    x_train, x_val, y_train, y_val = train_test_split(
        X, y, random_state=0)

    forest = RandomForestClassifier(random_state=0)
    forest.fit(x_train, y_train)
    y_pred = forest.predict(x_val)
    acc = forest.score(x_val, y_val)
    print('Accurary', acc)

def run():
    # get data
    train, test = get_data()

    # clean data
    train = clean_data(train)
    test = clean_data(test)

    # visualize data
    # heat_map(train, 'Pclass', 'Sex', 'Survived', 'mean')
    # heat_map(train, 'Sex', 'AgeBand', 'Survived', 'mean')

    # correlation matrix
    corr_matrix = train.corr()
    sns.heatmap(corr_matrix,
                xticklabels=corr_matrix.columns,
                yticklabels=corr_matrix.columns)
    # plt.show()

    # model training
    train_model(train)


if __name__ == '__main__':
    # pd.options.display.max_columns = 100
    run()


## NOTES FROM ANALYSIS ##

# females much more likely to survive than males
# higher Pclass (1 is highest, 3 is lowest) -> higher Survival % (R^2 = -0.3)
# higher Fare -> higher Survival % (R^2 = 0.3)



