import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df=pd.read_csv('tennis_stats.csv')

print(df.head())


##linear regression models that use two features to predict yearly earnings


features = df[['FirstServeReturnPointsWon']]
outcome = df[['Winnings']]

features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size = 0.8)


model = LinearRegression()
model.fit(features_train,outcome_train)


model.score(features_test,outcome_test)

prediction = model.predict(features_test)
plt.scatter(outcome_test,prediction, alpha=0.4)

features = df[['BreakPointsOpportunities']]
outcome = df[['Winnings']]

features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size = 0.8)


model = LinearRegression()
model.fit(features_train,outcome_train)


model.score(features_test,outcome_test)

prediction = model.predict(features_test)
plt.scatter(outcome_test,prediction, alpha=0.4)


features = df[['BreakPointsConverted']]
outcome = df[['Winnings']]

features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size = 0.8)


model = LinearRegression()
model.fit(features_train,outcome_train)


model.score(features_test,outcome_test)

prediction = model.predict(features_test)
plt.scatter(outcome_test,prediction, alpha=0.4)
plt.title('Linear Regression: Actual vs Predicted Winnings')
plt.xlabel('Actual Winnings')
plt.ylabel('Predicted Winnings')

plt.show()
plt.clf()



##linear regression models that use two features to predict yearly earnings

features = df[['BreakPointsOpportunities',
'FirstServeReturnPointsWon']]
outcome = df[['Winnings']]


features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size = 0.8)


model = LinearRegression()
model.fit(features_train,outcome_train)


model.score(features_test,outcome_test)

prediction = model.predict(features_test)
plt.scatter(outcome_test,prediction, alpha=0.4)


features = df[['BreakPointsSaved',
'TotalServicePointsWon']]
outcome = df[['Winnings']]


features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size = 0.8)


model = LinearRegression()
model.fit(features_train,outcome_train)


model.score(features_test,outcome_test)

prediction = model.predict(features_test)
plt.scatter(outcome_test,prediction, alpha=0.4)
plt.title('Linear Regression: Actual vs Predicted Winnings')
plt.xlabel('Actual Winnings')
plt.ylabel('Predicted Winnings')

plt.show()
plt.clf()


# linear regression models that use multiple features to predict yearly earnings

features = df[['FirstServe','FirstServePointsWon','FirstServeReturnPointsWon',
'SecondServePointsWon','SecondServeReturnPointsWon','Aces',
'BreakPointsConverted','BreakPointsFaced','BreakPointsOpportunities',
'BreakPointsSaved','DoubleFaults','ReturnGamesPlayed','ReturnGamesWon',
'ReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon',
'TotalServicePointsWon']]
outcome = df[['Winnings']]

features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size = 0.8)


model = LinearRegression()
model.fit(features_train,outcome_train)


model.score(features_test,outcome_test)

prediction = model.predict(features_test)
plt.scatter(outcome_test,prediction, alpha=0.4)

features = df[['FirstServe','FirstServePointsWon','FirstServeReturnPointsWon',
'SecondServePointsWon',
'BreakPointsSaved','DoubleFaults','ReturnGamesPlayed','ReturnGamesWon',
'ReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon',
'TotalServicePointsWon']]
outcome = df[['Winnings']]

features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size = 0.8)


model = LinearRegression()
model.fit(features_train,outcome_train)


model.score(features_test,outcome_test)

prediction = model.predict(features_test)
plt.scatter(outcome_test,prediction, alpha=0.4)

features = df[['FirstServe',
'SecondServePointsWon','SecondServeReturnPointsWon','Aces',
'BreakPointsConverted','BreakPointsFaced','BreakPointsOpportunities',
'BreakPointsSaved',
'TotalServicePointsWon']]
outcome = df[['Winnings']]

features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size = 0.8)


model = LinearRegression()
model.fit(features_train,outcome_train)


model.score(features_test,outcome_test)

prediction = model.predict(features_test)
plt.scatter(outcome_test,prediction, alpha=0.4)

plt.title('Linear Regression: Actual vs Predicted Winnings')
plt.xlabel('Actual Winnings')
plt.ylabel('Predicted Winnings')

plt.show()
plt.clf()




