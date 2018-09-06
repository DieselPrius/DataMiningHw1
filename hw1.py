import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pylab

# Make the graphs a bit prettier, and bigger
#pd.set_option('display.mpl_style', 'default')

# This is necessary to show lots of columns in pandas 0.12. 
# Not necessary in pandas 0.13.
pd.set_option('display.width', 5000) 
pd.set_option('display.max_columns', 60)

plt.rcParams['figure.figsize'] = (15, 5)


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

combine = [train_df, test_df]


################### TASK 3 ###############################
#Question 1 - What features are available in the dataset
print("=============== Question 1 ==================================")
print(set(list(train_df) + list(test_df)))
print("\n")


#Question 5 - which columns have null values?
print("=============== Question 5 ==================================")
print(train_df.isna().any())
print(test_df.isna().any())
print("\n")
#print(train_df[train_df["Embarked"].isnull()])


#Question 6 - what are the data types
print("=============== Question 6 ==================================")
print(train_df.dtypes)
print("\n")


#Question 7
print("=============== Question 7 ==================================")
#clean the data
#remove null ages
train_df = train_df[train_df["Age"].notnull()]
test_df = test_df[test_df["Age"].notnull()]
#remove null Fares
train_df = train_df[train_df["Fare"].notnull()]
test_df = test_df[test_df["Fare"].notnull()]

print("SUM")
print(train_df[["Age","SibSp","Parch","Fare"]].aggregate(sum))
print("\n")
print("MEAN")
print(train_df[["Age","SibSp","Parch","Fare"]].aggregate(np.mean))
print("\n")
print("STD")
print(train_df[["Age","SibSp","Parch","Fare"]].aggregate(np.std))
print("\n")
print(train_df[["Age","SibSp","Parch","Fare"]].aggregate(min))
print("\n")
print("25% percentile")
print(train_df[["Age","SibSp","Parch","Fare"]].quantile(0.25))
print("\n")
print("50% percentile")
print(train_df[["Age","SibSp","Parch","Fare"]].quantile(0.5))
print("\n")
print("75% percentile")
print(train_df[["Age","SibSp","Parch","Fare"]].quantile(0.75))
print("\n")
print("MAX")
print(train_df[["Age","SibSp","Parch","Fare"]].aggregate(max))
print("\n")


#question 8


#Question 9
print("=================== Question 9 ==================================")
print("Correlation Between Pclass and Survived:")
print(train_df["Pclass"].corr(train_df["Survived"]))
print(train_df[["Pclass","Survived"]].groupby("Pclass").aggregate(sum))
print("\n")


#Question 10
print("================== Question 10 ===============================")
print(train_df[["Sex","Survived"]].groupby("Sex").aggregate(sum))
print(197/93)
print("\n")


#Question 11
print("================== Question 11 ===============================")
ages = train_df[["Age","Survived"]]
ages.hist(bins=15, column="Age", by=ages["Survived"])  #########################################################################
print(ages)

#Question 12
print("================== Question 12 ===============================")
pclassAgeSur = pd.DataFrame(train_df[["Age","Pclass","Survived"]])
pclassAgeSur.hist(bins=15, column="Age", by=[pclassAgeSur["Pclass"],pclassAgeSur["Survived"]]) ###############################################

#Question 13
q13 = train_df[["Embarked","Sex","Fare","Survived"]]
#q13.hist(layout=(q13["Fare"],q13["Sex"]), by=[q13["Embarked"],q13["Survived"]])
print(train_df)


#Question 14
print("================== Question 14 ===============================")
#train_df[["Ticket","Survived"]][:100].hist(column="Ticket", by=train_df["Survived"])
q14 = pd.DataFrame(train_df["Ticket"])
#q14["Ticket"].value_counts()[:100].plot(kind="bar")
q14 = pd.DataFrame(train_df[["Ticket","Survived"]])
q14.Ticket = pd.to_numeric(q14.Ticket, errors='coerce').fillna(0).astype(np.int64)
q14 = q14[q14["Ticket"] != 0]
print("Ticket/Survival Correlation")
print(q14["Ticket"].corr(q14["Survived"]))


#Question 15
print("====================== Question 15 ===============================")
q15 = pd.DataFrame(train_df["Cabin"])
q15_2 = pd.DataFrame(test_df["Cabin"])
print(q15[q15["Cabin"].isnull()])
print(q15_2[q15_2["Cabin"].isnull()])

#Question 16
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
q16 = pd.DataFrame(train_df)
q16.loc[q16["Sex"] == "female", "Sex"] = 1
q16.loc[q16["Sex"] == "male", "Sex"] = 0
q16 = q16.rename(index=str, columns={"Sex":"Gender"})

test_df.loc[test_df["Sex"] == "female", "Sex"] = 1
test_df.loc[test_df["Sex"] == "male", "Sex"] = 0
test_df = test_df.rename(index=str, columns={"Sex":"Gender"})
print(q16)

#question 18
train_df = pd.DataFrame(q16)
print(train_df["Embarked"].value_counts())
train_df.loc[train_df["Embarked"].isnull(), "Embarked"] = "S"


#question 19
#print(train_df[train_df["Fare"].notnull()]["Fare"].agg(np.mode))
print(train_df[train_df["Fare"].notnull()]["Fare"].mode()[0])
train_df.loc[train_df["Fare"].isnull(), "Fare"] = train_df[train_df["Fare"].notnull()]["Fare"].mode()[0]



pylab.show()


