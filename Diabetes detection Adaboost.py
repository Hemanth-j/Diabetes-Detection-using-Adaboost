#diabities dataset bagging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
df=pd.read_csv("C:\\Users\\HEMANTH\\Downloads\\diabetes.csv")
"""sns.heatmap(df.isnull())
plt.show()"""
x=df.drop("Outcome",axis="columns")
"""x_corr=x.corr()
sns.heatmap(x_corr,annot=True)
plt.show()"""
y=df.Outcome
from sklearn.preprocessing import StandardScaler
scl=StandardScaler()
x_scl=scl.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_scl,y,train_size=0.8,stratify=y,random_state=0)
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(random_state=42)
"""clf.fit(x_train,y_train)
print(clf.score(x_train,y_train))
print(clf.score(x_test,y_test))"""
""""from sklearn.model_selection import cross_val_score
cv=cross_val_score(clf,x_scl,y,cv=5)
print(cv.mean())"""
from sklearn.ensemble import BaggingClassifier
bagg=BaggingClassifier(clf,n_estimators=100,max_samples=0.8,oob_score=True,random_state=0)
bagg.fit(x_train,y_train)
print(bagg.oob_score_)
print(bagg.score(x_train,y_train))
print(bagg.score(x_test,y_test))
from sklearn.model_selection import cross_val_score
cv=cross_val_score(bagg,x_scl,y,cv=5)
print("cross val score",cv.mean())
clf.fit(x_train,y_train)
print(clf.score(x_train,y_train))
print(clf.score(x_test,y_test))