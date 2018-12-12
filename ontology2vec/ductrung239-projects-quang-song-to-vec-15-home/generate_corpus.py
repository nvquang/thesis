

import pandas as pd
import random

import re
import string
import os

regex = re.compile('[%s]' % re.escape(string.punctuation))

result_folder = "./RESULT"


def cleaning_text(doc):
    
    # convert to lower case
    doc = doc.lower()

    # remove spacespace
    doc = re.sub(' +', ' ', doc)
    doc = doc.strip()
    doc = doc.replace(" ", "_")
    return doc


def read_data():
    df = pd.read_csv(
        "/floyd/input/dataset/songs_dataset_105k.csv")
    df.drop(columns=['ID', 'Md5 ', 'album_link', 'song_link', "Site"], inplace=True)

    values = {'Singer': 'Singer unknown', 
            'Song': "Song unknown", 
            'Author': "Author unknown", 
            'Album': "Album unknown", 
            'Genre': "Genre unknown"}

    df.fillna(value=values, inplace=True)

    print("nan: ", len(df[df.isnull().any(axis=1)]))

    df['Song'] = df['Song'].astype('str')
    df['Singer'] = df['Singer'].astype('str')
    df['Author'] = df['Author'].astype('str')
    df['Album'] = df['Album'].astype('str')
    df['Genre'] = df['Genre'].astype('str')

    df['Song'] = df['Song'].apply(cleaning_text)
    df['Singer'] = df['Singer'].apply(cleaning_text)
    df['Author'] = df['Author'].apply(cleaning_text)
    df['Album'] = df['Album'].apply(cleaning_text)
    df['Genre'] = df['Genre'].apply(cleaning_text)
    
    print("dataset info: ")
    print(df.info())

    print("dataset describle: ")
    print(df.describe())

    print("some first rows")
    print(df.head())

    return df 


def generate_corpus(df):
    ## Build a graph of song, singer, auther, album and genre
    # graph = { "a" : ["c"],
    #           "b" : ["c", "e"],
    #           "c" : ["a", "b", "d", "e"],
    #           "d" : ["c"],
    #           "e" : ["c", "b"],
    #           "f" : []
    #         }


    print("Generating graph ...")

    graph = {}

    print("Graph song")
    data_group_by_song = df.groupby(['Song']).groups
    for song in data_group_by_song:
        graph[song] = set()
        for row_index in data_group_by_song[song].values:
            graph[song].add(df.iloc[row_index]['Singer'])
            graph[song].add(df.iloc[row_index]['Author'])
            graph[song].add(df.iloc[row_index]['Album'])
            graph[song].add(df.iloc[row_index]['Genre'])
            

    print("Graph singer")
    data_group_by_singer = df.groupby(['Singer']).groups
    for singer in data_group_by_singer:
        graph[singer] = set()
        for row_index in data_group_by_singer[singer].values:
            graph[singer].add(df.iloc[row_index]['Song'])
            graph[singer].add(df.iloc[row_index]['Author'])
            graph[singer].add(df.iloc[row_index]['Album'])
            graph[singer].add(df.iloc[row_index]['Genre'])
            

    print("Graph author")
    data_group_by_author = df.groupby(['Author']).groups
    for author in data_group_by_author:
        graph[author] = set()
        for row_index in data_group_by_author[author].values:
            graph[author].add(df.iloc[row_index]['Song'])
            graph[author].add(df.iloc[row_index]['Singer'])
            graph[author].add(df.iloc[row_index]['Album'])
            graph[author].add(df.iloc[row_index]['Genre'])
            

    print("Graph album")
    data_group_by_album = df.groupby(['Album']).groups
    for album in data_group_by_album:
        graph[album] = set()
        for row_index in data_group_by_album[album].values:
            graph[album].add(df.iloc[row_index]['Song'])
            graph[album].add(df.iloc[row_index]['Singer'])
            graph[album].add(df.iloc[row_index]['Author'])
            graph[album].add(df.iloc[row_index]['Genre'])


    print("Graph genre")
    data_group_by_genre = df.groupby(['Genre']).groups
    for genre in data_group_by_genre:
        graph[genre] = set()
        for row_index in data_group_by_genre[genre].values:
            graph[genre].add(df.iloc[row_index]['Song'])
            graph[genre].add(df.iloc[row_index]['Singer'])
            graph[genre].add(df.iloc[row_index]['Author'])
            graph[genre].add(df.iloc[row_index]['Album'])


    print("Graph DONE")

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # generated random key
    nb_loop = 50
    for loop in range(0, nb_loop):
        print("Loop: ", loop)
        documents = []
        nb_of_docs = 1000000
        for i in range(0,nb_of_docs):
            if i % 100000 == 0:
                print("word: ", i)
                
            nb_of_word = random.randint(7, 10)
            words = []
            start_word = random.choice(list(graph.keys()))
            
            words.append(start_word)
            for word_index in range(0, nb_of_word):
                start_word = random.choice(list(graph[start_word]))
                words.append(start_word)
            documents.append(words)
        with open(result_folder + '/corpus_'+str(loop)+'.txt', 'w') as f:
            for item in documents:
                sentence = " ".join(item)
                f.write("%s\n" % sentence)
    print("Generated DONE")
        
    
if __name__ == '__main__':
    df = read_data()
    generate_corpus(df)
