import pandas as pd
import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary


df = pd.read_csv('app/content/media/dataset/iteung.csv', sep=';')

#membuat kamus
kamus_slang = open('app/content/media/dataset/slang_word.txt').read()
map_kamus_slang = {}
list_kamus_slang = []
for line in kamus_slang.split("\n"):
    if line != "":
        kamus = line.split("=")[0]
        kamus_luas = line.split("=")[1]
        list_kamus_slang.append(kamus)
        map_kamus_slang[kamus] = kamus_luas
list_kamus_slang = set(list_kamus_slang)

#memakai kamus
def perubahan_kata_slang(text):
    new_text = []
    
    for w in text.split():
        if w.upper() in list_kamus_slang:
            new_text.append(map_kamus_slang[w.upper()])
        else:
            new_text.append(w)
    return " ".join(new_text)

df['pesan'] = df['pesan'].apply(perubahan_kata_slang)

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocessing(prep):
    #Konvert ke huruf kecil
    prep = prep.lower()
    #Konvert www.* atau https?://* ke URL
    # prep = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',prep)
    # #Menghapus tambahan space
    # prep = re.sub('[\s]+', ' ', prep)
    # #Hanya Huruf
    # prep = re.sub("[^a-zA-Z]", " ", prep) 
    #Tokenisasi setiap kata
    token = nltk.word_tokenize(prep)
    #Stemming
    prep = [stemmer.stem(w) for w in token]
    #menggabungkan kembali kata ke dalam satu string dengan dipisahkan spasi.
    return " ".join(prep)

df['clean'] = df['pesan'].apply(preprocessing)


from sklearn.feature_extraction.text import TfidfVectorizer 

tf_idf_vectorizer = TfidfVectorizer(use_idf=True,ngram_range=(1,2))
final_data = tf_idf_vectorizer.fit_transform(df['clean'])
final_data

from sklearn.model_selection import train_test_split
y, levels = pd.factorize(df['label'])
X_train, X_test, y_train, y_test = train_test_split(final_data, y, test_size=0.2, random_state=2) 


from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

model_naive = MultinomialNB().fit(X_train, y_train) 
predicted_naive = model_naive.predict(X_test)



text_clf_nb = Pipeline([('tfidf', TfidfVectorizer(use_idf=True,ngram_range=(1,2))), ('clf', MultinomialNB()),])
a = df['clean']
b = df['label']
text_clf_nb.fit(a, b)

import pickle
pickle.dump(text_clf_nb, open('my_model.sav', 'wb'))

def prediksi(kalimat):
    import pickle
    x = [kalimat]
    text_clf_nb = pickle.load(open('my_model.sav', 'rb'))
    predictions = text_clf_nb.predict(x)
    print(predictions)

prediksi("maaf pak sebelumnya, kok nama saya belum terabsen ya ?")