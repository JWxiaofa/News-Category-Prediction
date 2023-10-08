import re
from collections import Counter
import json
import pprint
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from sklearn.model_selection import train_test_split


file = open('News_Category_Dataset_v3.json')
data = []
for line in file.readlines():
    dic = json.loads(line)
    data.append(dic)

lst_stopwords = nltk.corpus.stopwords.words("english")

def preprocess_text(text: str, stemm=False, lemm=False, stop_words=False):

    text = text.lower()
    text = re.sub(r'[^A-Za-z ]+', '', text)

    lst_text = text.split()

    if stop_words:
        lst_text = [word for word in lst_text if word not in lst_stopwords]

    if stemm:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]

    if lemm:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]

    text = " ".join(lst_text)
    return text

count = Counter()
for news in data:
    count[news['category']] += 1

del_lst = []
for key in count.keys():
    if count[key] < 5000:
        del_lst.append(key)
del_lst.append('QUEER VOICES')

for cat in del_lst:
    del count[cat]

for i in range(len(data)-1, -1, -1):
    if data[i]['category'] in del_lst:
        data.pop(i)

for news in data:
    del news['link']
    del news['authors']
    del news['date']
    news['text'] = news['headline'] + ' ' + news['short_description']
    del news['short_description']
    del news['headline']

    news['text'] = preprocess_text(news['text'], stemm=True, lemm=True, stop_words=True)


train, test = train_test_split(data, test_size=0.2)
test, dev = train_test_split(test, test_size=0.5)

resultFile = open('train_data.py', 'w')
resultFile.write('data = ' + pprint.pformat(train))

resultFile = open('dev_data.py', 'w')
resultFile.write('data = ' + pprint.pformat(dev))

resultFile = open('test_data.py', 'w')
resultFile.write('data = ' + pprint.pformat(test))

