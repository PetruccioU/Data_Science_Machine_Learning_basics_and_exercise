import pandas as pd
import matplotlib.pyplot as plt
dat = pd.read_csv('Titanic-Dataset.csv')

#FILTERING
print('------')
print('FILTERING')
print('------')

print(dat)
print(dat.info())
print(dat.shape)
print(dat['Name'])
#print(type(dat['Name']))
print(dat[['Name', 'Age']].head(10))
print(dat.loc[[5,10,15],['Age','Cabin']])
print(dat.iloc[[5,10,15],[1,2]])
print(dat.iloc[5:11,:3]) # [:3]=[0:3]
print(dat[dat['Age']>60])
print(dat['Age'] > 15)
print(type(dat['Age'] > 15))
print(dat[dat['Age'].isin([5,10,15])])  # dat[dat['Age'].isin([5,10,15])]) в isin могут быть колонки из другово датафрейма
print('------')
print(dat[(dat['PassengerId'] == 5) | (dat['PassengerId'] == 10)]) # | это "ИЛИ", & это "И"
print(dat[dat['Name'].notna()])   #без пропусков по имени
print(dat.loc[dat['Name'].notna() & dat['Age'].notna(),'Age'])  # Вывести имена Без пропусков по Имени и Возрасту
#print(dat[[dat['Name'].notna()],[dat['Age']]])

# SORTING
print('------')
print('SORTING')
print('------')
#print(dat.sort_values('Age'))
print(dat[['Name', 'Age']].sort_values('Age')) # sort by age, showing name and age
#print(dat.sort_values(['Age','Name'],ascending=[False,True]).head(10))
print(dat[['Name', 'Age']].sort_values(['Age','Name'],ascending=[False,True]).head(10))

# COPY
print('------')
print('COPY')
print('------')
dat2 = dat.copy(deep=True) # true copy of every element of dat into dat2, not just a link to dat

#CONCATINATION
print('------')
print('CONCATINATION(summing)')
print('------')

dat3 = pd.concat([dat,dat2])    # сложение по строкам, просто сначала первый массив, потом второй, столбцы те-же
print(dat3)
dat4 = pd.concat([dat,dat2],axis=1)   # Сложение по столбцам(было 12 столбцов станет 24) добавляем справа второй массив
print(dat4)

#JOIN
print('------')
print('JOIN')
print('------')
newdf = pd.DataFrame(index=dat.index)
newdf['PassengerId']=dat['PassengerId']
newdf['IDisEven']=dat['PassengerId'].apply(lambda x: x % 2 == 0)
print(newdf)
JoinDat = pd.merge(newdf,dat,how='inner')
print(JoinDat)
print(dat.count())
print(dat['Age'].count())
print(dat['Age'].mean(),dat['Age'].median())
print(dat[['Age','Survived']].describe())

#GROUP BY
print('------')
print('GROUP BY')
print('------')
print(dat.groupby('Sex')['Age'].mean(),dat.groupby('Sex')['Age'].median())
print(dat.groupby('Sex')['Age'].describe())
print(dat.groupby(['Sex', 'Survived'])['Age'].agg(['mean', 'median'])) # группируем по полу и выживанию, выводим среднее и медиану
print(dat['Sex'].value_counts())
print(dat.groupby('Sex')['Sex'].count())

#CORRELATION
print('------')
print('CORRELATION')
print('------')
print(dat.corr())

#PLOTTING
print('------')
print('PLOTTING')
print('------')

dat['Age'].plot(kind='hist', bins = 20)
plt.show()

dat['Age'].plot(kind='kde', xlim=[0, 100])
plt.show()

dat.groupby('Sex')['Age'].plot(kind='kde', xlim=[0, 100], legend=True)
plt.show()

#CHANGING
print('------')
print('CHANGING')
print('------')

tmp_dat = dat.copy(deep=True)
tmp_dat['Pclass'] = 1
print(tmp_dat['Pclass'].value_counts())
tmp_dat['IsAdult'] = tmp_dat['Age'] >= 18  # tmp_dat['Age'] >= 18 - Булевая маска
tmp_dat['IsAdultSurvived'] = tmp_dat['IsAdult']
tmp_dat.loc[tmp_dat['Survived'] == 0, 'IsAdultSurvived'] = False
print(tmp_dat[['IsAdultSurvived','Age','Survived']].head(50))
print(tmp_dat.rename(
    columns={'IsAdultSurvived': 'AdultSurvived'}
).head(1))
print(tmp_dat.head(5))
print(tmp_dat.rename(
    columns=str.lower
).head(1))
tmp_dat['lower_case_name'] = tmp_dat['Name'].apply(lambda x: x.lower())
print(tmp_dat['lower_case_name'])

tmp_dat['lower_case_name2'] = tmp_dat['Name'].str.lower()
print(tmp_dat['lower_case_name2'])