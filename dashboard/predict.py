import pandas as pd
import pickle
import re
import os
import nltk
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import json
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from django.conf import settings
from .models import DatasetModel, PrediksiDataModel
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from pycm import ConfusionMatrix
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud



# get data from database\
def retrain(data_df_pesan, data_df_label):
    pd.set_option('display.max_colwidth', None)
    data = pd.DataFrame(data_df_pesan)
    data2 = pd.DataFrame(data_df_label)
    list_pesan = data['pesan'].tolist()
    list_label = data2['label'].tolist()
    df = pd.DataFrame({'pesan':list_pesan, 'label':list_label})

    tf_idf_vectorizer = TfidfVectorizer(use_idf=True,ngram_range=(1,2))
    final_data = tf_idf_vectorizer.fit_transform(df['pesan'])
    final_data

    y, levels = pd.factorize(df['label'])
    X_train, X_test, y_train, y_test = train_test_split(final_data, y, test_size=0.2, random_state=2) 
    model_naive = MultinomialNB().fit(X_train, y_train) 
    predicted_naive = model_naive.predict(X_test)
    
    plt.figure(dpi=500, facecolor='#25282c')
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#25282c')
    mat = pd.crosstab(levels[y_test], levels[predicted_naive], dropna = False)
    heatmap = sns.heatmap(mat.T, annot=True, ax=ax, fmt="d", cbar="whitesmoke")
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=30, color="whitesmoke")
    heatmap.set_yticklabels(heatmap.get_yticklabels(), color="whitesmoke")
    plt.rcParams['text.color'] = 'whitesmoke'

    plt.title('Confusion Matrix', color="whitesmoke")
    plt.xlabel('true label', color="whitesmoke")
    plt.ylabel('predicted label', color="whitesmoke")
    base_dir = settings.BASE_DIR
    file_path_cm = os.path.join(base_dir, 'content/assets/img/confusion_matrix.png')
    plt.savefig(file_path_cm, bbox_inches='tight')
    plt.close()

    text_clf_nb = Pipeline([('tfidf', TfidfVectorizer(use_idf=True,ngram_range=(1,2))), ('clf', MultinomialNB()),])
    a = df['pesan']
    b = df['label']
    text_clf_nb.fit(a, b)
    file_path_pickle = os.path.join(settings.MEDIA_ROOT, 'dataset/my_model.sav')
    pickle.dump(text_clf_nb, open(file_path_pickle, 'wb'))

def wordCloudMarah(data_df_pesan, data_df_label):
    pd.set_option('display.max_colwidth', None)
    data = pd.DataFrame(data_df_pesan)
    data2 = pd.DataFrame(data_df_label)
    list_pesan = data['pesan'].tolist()
    list_label = data2['label'].tolist()
    df = pd.DataFrame({'pesan':list_pesan, 'label':list_label})
    marah = ' '.join([text for text in df['pesan'][df['label'] == 'marah']])
    wordcloud = WordCloud(width=800, height=500, random_state=21,
            max_font_size=110,background_color="rgba(255, 255, 255, 0)"
            , mode="RGBA").generate(marah)
    plt.figure(dpi=600)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title("kata yang sering digunakan dalam label marah")
    base_dir = settings.BASE_DIR
    file_path = os.path.join(base_dir, 'content/assets/img/wordCloud_marah.png')
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()

def wordCloudSenang(data_df_pesan, data_df_label):
    pd.set_option('display.max_colwidth', None)
    data = pd.DataFrame(data_df_pesan)
    data2 = pd.DataFrame(data_df_label)
    list_pesan = data['pesan'].tolist()
    list_label = data2['label'].tolist()
    df = pd.DataFrame({'pesan':list_pesan, 'label':list_label})
    senang = ' '.join([text for text in df['pesan'][df['label'] == 'senang']])
    wordcloud = WordCloud(width=800, height=500, random_state=21,
            max_font_size=110,background_color="rgba(255, 255, 255, 0)"
            , mode="RGBA").generate(senang)
    plt.figure(dpi=600)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title("kata yang sering digunakan dalam label senang")
    base_dir = settings.BASE_DIR
    file_path = os.path.join(base_dir, 'content/assets/img/wordCloud_senang.png')
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()

def wordCloudSedih(data_df_pesan, data_df_label):
    pd.set_option('display.max_colwidth', None)
    data = pd.DataFrame(data_df_pesan)
    data2 = pd.DataFrame(data_df_label)
    list_pesan = data['pesan'].tolist()
    list_label = data2['label'].tolist()
    df = pd.DataFrame({'pesan':list_pesan, 'label':list_label})
    sedih = ' '.join([text for text in df['pesan'][df['label'] == 'sedih']])
    wordcloud = WordCloud(width=800, height=500, random_state=21,
            max_font_size=110,background_color="rgba(255, 255, 255, 0)"
            , mode="RGBA").generate(sedih)
    plt.figure(dpi=600)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title("kata yang sering digunakan dalam label sedih")
    base_dir = settings.BASE_DIR
    file_path = os.path.join(base_dir, 'content/assets/img/wordCloud_sedih.png')
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()

def wordCloudPercaya(data_df_pesan, data_df_label):
    pd.set_option('display.max_colwidth', None)
    data = pd.DataFrame(data_df_pesan)
    data2 = pd.DataFrame(data_df_label)
    list_pesan = data['pesan'].tolist()
    list_label = data2['label'].tolist()
    df = pd.DataFrame({'pesan':list_pesan, 'label':list_label})
    percaya = ' '.join([text for text in df['pesan'][df['label'] == 'percaya']])
    wordcloud = WordCloud(width=800, height=500, random_state=21,
            max_font_size=110,background_color="rgba(255, 255, 255, 0)"
            , mode="RGBA").generate(percaya)
    plt.figure(dpi=600)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title("kata yang sering digunakan dalam label percaya")
    base_dir = settings.BASE_DIR
    file_path = os.path.join(base_dir, 'content/assets/img/wordCloud_percaya.png')
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()

def wordCloudTakut(data_df_pesan, data_df_label):
    pd.set_option('display.max_colwidth', None)
    data = pd.DataFrame(data_df_pesan)
    data2 = pd.DataFrame(data_df_label)
    list_pesan = data['pesan'].tolist()
    list_label = data2['label'].tolist()
    df = pd.DataFrame({'pesan':list_pesan, 'label':list_label})
    takut = ' '.join([text for text in df['pesan'][df['label'] == 'takut']])
    wordcloud = WordCloud(width=800, height=500, random_state=21,
            max_font_size=110,background_color="rgba(255, 255, 255, 0)"
            , mode="RGBA").generate(takut)
    plt.figure(dpi=600)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title("kata yang sering digunakan dalam label takut")
    base_dir = settings.BASE_DIR
    file_path = os.path.join(base_dir, 'content/assets/img/wordCloud_takut.png')
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()


def coba(data_df_pesan, data_df_label):
    pd.set_option('display.max_colwidth', None)
    data = pd.DataFrame(data_df_pesan)
    data2 = pd.DataFrame(data_df_label)
    list_pesan = data['pesan'].tolist()
    list_label = data2['label'].tolist()
    df = pd.DataFrame({'pesan':list_pesan, 'label':list_label})

    tf_idf_vectorizer = TfidfVectorizer(use_idf=True,ngram_range=(1,2))
    final_data = tf_idf_vectorizer.fit_transform(df['pesan'])
    final_data

    y, levels = pd.factorize(df['label'])
    X_train, X_test, y_train, y_test = train_test_split(final_data, y, test_size=0.2, random_state=2) 
    model_naive = MultinomialNB().fit(X_train, y_train) 
    predicted_naive = model_naive.predict(X_test)

    cm1=ConfusionMatrix(y_test,predicted_naive,digit=2)
    cm1.relabel(mapping={0:"marah",1:"senang",2:"takut",3:"sedih",4:"percaya"})
    cm1.save_csv(os.path.join(settings.MEDIA_ROOT,"dataset/evaluasi"),class_param=["ACC","FP","PPV","ERR","TNR","PRE","TPR"])
    df = pd.read_csv(os.path.join(settings.MEDIA_ROOT,'dataset/evaluasi.csv'), sep=',')
    json_records = df.reset_index().to_json(orient ='records')
    data_json = []
    data_json = json.loads(json_records)

    score_naive1 = accuracy_score(predicted_naive, y_test)
    akurasi = '%.2f' % (score_naive1*100)

    #df_html = df.to_html()
    return(data_json, akurasi)


# prediksi
def prediksi(kalimat):
    x = [kalimat]
    file_path = os.path.join(settings.MEDIA_ROOT, 'dataset/my_model.sav')
    text_clf_nb = pickle.load(open(file_path, 'rb'))
    predictions = text_clf_nb.predict(x)
    prediction = ''.join(predictions)
    return(prediction)

# # def read_data():
# #     engine = create_engine('postgresql+psycopg2://postgres:baktiqilan@/semi')
# #     data = pd.read_sql_query('select pesan, label from "dashboard_datasetmodel"', con=engine)
# #     print(data)
file_path = os.path.join(settings.MEDIA_ROOT, 'dataset/slang_word.txt')
kamus_slang1 = open(file_path).read()
map_kamus_slang = {}
list_kamus_slang = []
for line in kamus_slang1.split("\n"):
    if line != "":
        kamus = line.split("=")[0]
        kamus_luas = line.split("=")[1]
        list_kamus_slang.append(kamus)
        map_kamus_slang[kamus] = kamus_luas
list_kamus_slang = set(list_kamus_slang)

# kamus slang
def kamus_slang(text):
    new_text = []
    for w in text.split():
        if w.upper() in list_kamus_slang:
            new_text.append(map_kamus_slang[w.upper()])
        else:
            new_text.append(w)
    return " ".join(new_text)


factory = StemmerFactory()
stemmer = factory.create_stemmer()

# preprocess
def preprocess(text):
    text = text.lower()

    text = re.sub('((www\\.[^\\s]+)|(https?://[^\\s]+))','URL',text)
    text = re.sub('[\\s]+', ' ', text)
    text = re.sub("[^a-zA-Z]", " ", text) 
    token = nltk.word_tokenize(text)
    text = [stemmer.stem(w) for w in token]
    #text = [stemmer.stem(w) for w in text]
    return " ".join(text)


def prediksi_list(kalimat):
    file_path = os.path.join(settings.MEDIA_ROOT, 'dataset/my_model.sav')
    text_clf_nb = pickle.load(open(file_path, 'rb'))
    predictions = text_clf_nb.predict(kalimat)
    return(predictions)


