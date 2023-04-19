import os.path
import pandas as pd
import sklearn.metrics._plot.precision_recall_curve
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
nltk.download('punkt')
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
from matplotlib import pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay
import numpy as np
from sklearn.model_selection import GridSearchCV


#print(os.getcwd())
df = pd.read_csv('labeled.csv', sep=",")    #  считали дата Фрэйм из csv файла
df["toxic"] = df["toxic"].apply(int)  # преобразовали код токсичности в int
#print(df.head(5))                     # вывести первые пять элементов массива
#print(df["toxic"].value_counts())       # проверим распределение таксичных коментов(сколько их там)
#print(type(df))
#print("------------")
#for c in df[df["toxic"] == 1]["comment"].head(5):   # посмотреть 5 токсичных комментов
#    print(c)
#print("------------")
#for c in df[df["toxic"] == 0]["comment"].head(5):   # посмотреть 5 обычных комментов
#    print(c)

train_df, test_df = train_test_split(df,test_size=500)    # возьмем для тестирования 500, а для обучения оставшиеся
#print(test_df.shape)                                      # проверим количество элементов в массиве для теста
#print(type(test_df))
#print(test_df["toxic"].value_counts())

# Будем анализировать выборку моделью LogisticRegression из sklearn.linear_model
# Модель принимает вещественные векторы, поэтому нужно преобразовать текст в численные векторы

# Проведем пред обработку текста:
# 1) разбить текст на токены
# 2) удалить токены(слова) которые не несут смысловой нагрузки: знаки пунктуации, междометия
# 3) применить к каждому токену стэмминг(удалить окончания, приставки, привести к нижнему регистру)
# 4) создание векторов

#sentence_example = df.iloc[1]["comment"]               # выбираем строки с комментами
#token = word_tokenize(sentence_example,language="russian")   # токенизируем строки т.е разделяем строки на слова
#token_without_punctuation = [i for i in token if i not in string.punctuation]   # делаем список токенов без пунктуации
#russian_stop_words = stopwords.words("russian")  # создаем список со стоп словами
#token_without_stopwards_and_punctuation = [i for i in token_without_punctuation if i not in russian_stop_words]
## создаем список с токенами без пунктуации и без стоп слов
#snowball = SnowballStemmer(language="russian")    # создаем обьект класса стэммер
#stemmed_tokens = [snowball.stem(i) for i in token_without_stopwards_and_punctuation] # применяем стэммер к нашим
## токенам (переводим в нижний регистр и удаляем окончания)
#print(stemmed_tokens)

russian_stop_words = stopwords.words("russian")
snowball = SnowballStemmer(language="russian")

def tokinize_sentence(sentence: str, remove_stop_words: bool == True):
    tokens = word_tokenize(sentence, language="russian")
    tokens = [i for i in tokens if i not in string.punctuation]
    if remove_stop_words:
        tokens = [i for i in tokens if i not in russian_stop_words]
    tokens = [snowball.stem(i) for i in tokens]
    return tokens

# После того как у нас готов массив токенов, можем применять TFIDF векторизацию
# Приступаем к векторизации, создаем векторайзер:

#vectorizer = TfidfVectorizer(tokenizer=lambda x: tokinize_sentence(x,remove_stop_words=True))
## Обучаем векторайзер: после того мы его обучили готовые фичи можно передавать в модель Машинного обучения
#features = vectorizer.fit_transform(train_df["comment"])
#
## Начинаем обучать нашу модель логистической регрессии
#model = LogisticRegression(random_state=0)  # Создали обьект класса регрессии
#model.fit(features, train_df["toxic"])      # используем метов fit, чтобы анализировать
#LogisticRegression(random_state=0)
#print(model.predict(features[0]))
#print(train_df["comment"].iloc[0])

# Но хотелось бы скармливать модели не векторы а текст.. В SKlearn существует класс pipeline
model_pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(tokenizer=lambda x: tokinize_sentence(x, remove_stop_words=True))),
    ("model", LogisticRegression(random_state=0))
]
)
model_pipeline.fit(train_df["comment"], train_df["toxic"])        # обучаем модель массивами данных
print("Предскажем комментарий:Привет, у меня все хорошо. И его вероятность:")
print(model_pipeline.predict(["Привет, у меня все хорошо!"]))       # скормим модели строка на предсказание
print(model_pipeline.predict_proba(["Привет, у меня все хорошо!"]))
print("Предсказали..")
#print("Ты пидор!")
#print(model_pipeline.predict(["Ты пидор!"]))             # скормим модели строка на предсказание
print(precision_score(y_true=test_df["toxic"], y_pred=model_pipeline.predict(test_df["comment"])))
print(recall_score(y_true=test_df["toxic"], y_pred=model_pipeline.predict(test_df["comment"])))
prec, rec, thresholds = precision_recall_curve(y_true=test_df["toxic"], probas_pred=model_pipeline.predict_proba(test_df["comment"])[:, 1])
#plot_precision_recall_curve(estimator=model_pipeline, x=test_df["comment"], y=test_df["toxic"])
#_plot.precision_recall_curve.PrecisionRecallDisplay(precision=prec, recall=rec)
print(np.where(prec > 0.95))    # распечатаем массив с индексами элементов где точность(prec) > 0.95
disp = PrecisionRecallDisplay(precision=prec, recall=rec)
disp.plot()
plt.show()
print("thresholds:")
print(thresholds[450])
print("precision_score(threshold>450):")
print(precision_score(y_true=test_df["toxic"], y_pred=model_pipeline.predict_proba(test_df["comment"])[:, 1] > thresholds[450]))
print("recall_score(threshold>450):")
print(recall_score(y_true=test_df["toxic"], y_pred=model_pipeline.predict_proba(test_df["comment"])[:, 1] > thresholds[450]))
# Base line: основное условие чтобы precision была не менее 0.95 выполнено
print('Grid model:')


# Улучшим модель, попробуем подобрать "Гипер параметры" исп класс GridSearchCV
# из sklearn он перебирает разные параметры на кросс валидации
# Logistic обернем в GreedSearchCV
grid_pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(tokenizer=lambda x: tokinize_sentence(x, remove_stop_words=True))),
    ("model",
     GridSearchCV(
         LogisticRegression(random_state=0),
         param_grid={'C':[0.1,1,10.]},          # параметр регуляризации функция запускается с 3я разными параметрами param_grid 0.1 1 10 (дефолтный 1)
         cv=3,       # во время кросс валидации(cross validation) будем разбивать на 3 фолда
         verbose=4   # выводим максимальную информацию
                  ))
]
)
grid_pipeline.fit(train_df["comment"], train_df["toxic"])   # Обучаем новую модель(Грид)
# Выбрали коэффициент С = 10, у него самыя высокая точность [CV 3/3] END ...C=10.0;, score=0.870 total time=   0.5s
model_pipeline_c10 = Pipeline([
    ("vectorizer", TfidfVectorizer(tokenizer=lambda x: tokinize_sentence(x, remove_stop_words=True))),
    ("model", LogisticRegression(random_state=0,C=10))
]
)
model_pipeline_c10.fit(train_df["comment"], train_df["toxic"])
prec_c10, rec_c10, thresholds_c10 = precision_recall_curve(y_true=test_df["toxic"], probas_pred=model_pipeline_c10.predict_proba(test_df["comment"])[:, 1])
print(np.where(prec_c10 > 0.95))
disp_c10 = PrecisionRecallDisplay(precision=prec_c10, recall=rec_c10)
disp_c10.plot()
plt.show()
print("thresholds_c10:")
print(thresholds_c10[440])
print("c10_precision_score(threshold>440):")
print(precision_score(y_true=test_df["toxic"], y_pred=model_pipeline_c10.predict_proba(test_df["comment"])[:, 1] > thresholds_c10[440]))
print("c10_recall_score(threshold>440):")
print(recall_score(y_true=test_df["toxic"], y_pred=model_pipeline_c10.predict_proba(test_df["comment"])[:, 1] > thresholds_c10[440]))
# Готово!
# библиотеки основанные на нейросетях: transformers, spacy (для nlp)
# embeding - что это?

