from ast import literal_eval
import os
import pandas as pd
from gensim.models import Word2Vec
import multiprocessing
from bs4 import BeautifulSoup
import re
import nltk
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import string



def replace_HTML_tags(df):
    count = 0
    title = df.Title.astype(str)
    df.Title = ([words.replace('&lt;', '<').replace('lt;', '<').replace('&gt;', '>').replace('gt;', '>').replace('&#xA;', '\n').replace('&quot;', '').replace(':', '').replace('&apos;','').replace(
    '&amp;', '') for words in title])
    body = df.Body.astype(str)
    df.Body = ([words.replace('&lt;', '<').replace('lt;', '<').replace('&gt;', '>').replace('gt;', '>').replace('&quot;', '').replace(':', '').replace('&apos;','').replace(
    '&amp;', '') for words in body])
    count += 1
    if (count % 100 == 0):
        print("Processing ", count, " th post")

def remove_code(df):
    code = []
    no_code = []
    for idx, row in df.iterrows():
        qid = int(row.get('Id'))
        if qid != "":
            body = row.get("Body")
            match_blockquote = re.findall(r'((?<=<pre><code>)(.*?)(?<=</code></pre>))', body)
            file_lst_trimmed = [str(file[0]) for file in match_blockquote]
            print(int(row.get('Id')), file_lst_trimmed)
            if(file_lst_trimmed == []):
                file_lst_trimmed = 'Xox'
            code.append(file_lst_trimmed)


            code_match_str = re.compile(r'((?<=<pre><code>)(.*?)(?<=</code></pre>))', re.S)
            no_code_select = code_match_str.sub('', body)
            soup = BeautifulSoup(no_code_select, 'lxml')
            refined_body = soup.get_text()


            no_code.append(refined_body)

    df['No_code'] = no_code
    return df

def lematize(POS_tag):
    wordnet_lemmatizer = WordNetLemmatizer()

    adjective_tags = ['JJ', 'JJR', 'JJS']

    lemmatized_text = []

    for word in POS_tag:
        if word[1] in adjective_tags:
            lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(word[0], pos="a")))
        else:
            lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(word[0])))  # default POS = noun


    POS_tag = nltk.pos_tag(lemmatized_text)

    return lemmatized_text

def stopWordRemoval(POS_tag):
    stopwords = []

    wanted_POS = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'FW']

    for word in POS_tag:
        if word[1] not in wanted_POS:
            stopwords.append(word[0])

    punctuations = list(str(string.punctuation))

    stopword_file = open("long_stopwords.txt", "r")

    lots_of_stopwords = []

    for line in stopword_file.readlines():
        lots_of_stopwords.append(str(line.strip()))

    stopwords_plus = punctuations + lots_of_stopwords + stopwords
    stopwords_plus = set(stopwords_plus)

    return stopwords_plus


def collocationGeneration(df):
    collocations_list = []

    for idx, row in df.iterrows():
        text1 = row.No_code
        text2 = row.Title
        space = ' '
        text = text2 + space + text1

        text = re.sub(r"[^a-zA-Z0-9]", " ", text)
        text = text.lower()
        text = word_tokenize(text)
        POS_tag = nltk.pos_tag(text)
        print("Tokenized Text with POS tags: \n")
        lemmatized_text = lematize(POS_tag)

        stopword_list = stopWordRemoval(POS_tag)
        processed_text = []
        for word in lemmatized_text:
            if word not in stopword_list:
                processed_text.append(word)


        #collocation generation
        collocations = []
        collocation = " "
        for word in processed_text:

            if word in stopword_list:
                if collocation != " ":
                    collocations.append(str(collocation).strip().split())
                collocation = " "
            elif word not in stopword_list:
                collocation += str(word)
                collocation += " "

        collocations_list.append(collocations)

    df['Collocations'] = [lst for lst in collocations_list]


    return df
  
def main():

    path_dir_raw = os.path.dirname('filepath')
    read_file_raw = os.path.join(path_dir_raw, 'DP_AP_specific_tagList_Filtered_V5.1.csv')

    # Read DP data
    dp_data = pd.read_csv(read_file_raw, sep='\t')

    # preprocessing
    replace_HTML_tags(dp_data)
    data = remove_code(dp_data)

    #collocation generation
    data_with_collocations = collocationGeneration(data)

    ''' Generate keyterms '''
    collocations = data_with_collocations.Collocations
    collocation_list = collocations.tolist()
    collocation_list = [literal_eval(post) for post in collocation_list]

    words = data_with_collocations.No_code_tokens
    words_list = words.tolist()
    words_list = [literal_eval(post) for post in words_list]

    keyTerms = []
    for a, b in zip(collocation_list, words_list):
        if (a != None) and (b != None):
            comb = a + b
            keyTerms.append(comb)

    ''' Train word2vec models '''
    cores = multiprocessing.cpu_count()
    RL_model = Word2Vec(keyTerms, min_count=1, size=200, workers=cores - 1, hs=0, window=3, sg=1, iter=100)
    print('Skipgram created with a vocab size of: ', len(RL_model.wv.vocab))

    path_to_save = (
        'RL_Model.model')

    RL_model.save(path_to_save)
  
  
