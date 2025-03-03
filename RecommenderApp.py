"""
HONOUR CODE:
This project is an adaptation of my previous project I did for the seminar
Statistical Language Processing at the University of Tübingen in the Summer
Semester 2024. The code here is a result of my own work on the patterns given
by my lecturers (they include method declarations and docstring descriptions)
and my later work to adapt the project, including the creation of the class
RecommenderApp, creation of the __init__ function for the class, adaptation of
the existing methods for the class, as well as the adaptation for the command line use.

Szymon Tomasz Kossowski
"""
import csv
import argparse

from gensim.models import KeyedVectors
import spacy
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd

#
# -- Important Note --
# Several functions have been deprecated in the
# latest scipy package (1.13).
# Since the gensim package depends on scipy, you will need to
# install an earlier version of scipy (1.11.4)
#

# # ToDo: load the spacy english small model,
# #  disable parser and ner pipes
# print('Loading spaCy...')
# nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
#
# # ToDo: load word2vec embeddings,
# #  which are located in the same directory as this script.
# #  Limit the vocabulary to 100,000 words
# #  - should be enough for this application
# #  - loading all takes a long time
# print('Loading word2vec embeddings...')
# emb = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True, limit=100000)

def cosine_similarity(v1, v2):
    """
    Calculate the cosine similarity of v1 and v2.

    :param v1: vector 1
    :param v2: vector 2
    :return: cosine similarity
    """
    return dot(v1, v2) / (norm(v1) * norm(v2))


class RecommenderApp:
    def __init__(self, spacy_model, embeddings, data_file):
        self.nlp = spacy.load(spacy_model, disable=['parser', 'ner'])
        self.emb = KeyedVectors.load_word2vec_format(embeddings, binary=True, limit=100000)
        self.data = self.load_data(data_file)
        self.preprocess_texts()
        self.get_vectors()

    def load_data(self, filename):
        """
        Load the Ted Talk data from filename and extract the
        "description" and "url" columns. Return a dictionary of dictionaries,
        where the keys of the outer dictionary are unique integer values which
        will be used as IDs.
        Each inner dictionary represents a row in the input file and contains
        the keys 'description', and 'url'.

        :param filename: input filename in csv format
        :return: dict of dicts, where the inner dicts represent rows in the input
        file, and the keys in the outer dict serve as IDs.
        """
        data = {}
        with open(filename, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row_id, row in enumerate(reader, start=0):
                data[row_id] = {'description': row['description'], 'url': row['url']}
        return data


    def preprocess_text(self, text):
        """
        Preprocess one text. Helper function for preprocess_texts().

        Preprocessing includes lowercasing and removal of stopwords,
        punctuation, whitespace, and urls.

        The returned list of tokens could be an empty list.

        :param text: one text string
        :return: list of preprocessed token strings. May be an empty list if all tokens are eliminated.
        """
        text = text.lower()  # lowercasing the text
        doc = self.nlp(text)  # launching the model
        # returning only these tokens that meet the criteria
        return [token.text for token in doc if
                not token.is_stop and not token.is_punct and not token.is_space and not token.like_url]


    def preprocess_texts(self):
        """
        Preprocess the description in each inner dict of data_dict by
        lowercasing and removing stopwords, punctuation, whitespace, and urls.
        The list of token strings for an individual text is not a set,
        and therefore may contain duplicates. Add a new key 'pp_text'
        to each inner dict, where the value is a list[str] representing
        the preprocessed tokens the text.

        :param data_dict: a nested dictionary with a structure as returned by load_data()
        """
        for inner_dict in self.data.values():  # iteration through inner dicts
            # creation of new entries with preprocessed description as a value
            inner_dict["pp_text"] = self.preprocess_text(inner_dict["description"])


    def get_vector(self, tokens):
        """
        Calculate a single vector for the preprocessed word strings in tokens.
        The vector is calculated as the mean of the word2vec vectors for the
        words in tokens. Words that are not contained in the word2vec pretrained
        embeddings are ignored. If none of the tokens are contained in word2vec,
        return None.

        :param tokens: list of strings containing the preprocessed tokens
        :return: mean of the word2vec vectors of the words in tokens, or None
        if none of the tokens are in word2vec.
        """
        valid_tokens = [token for token in tokens if token in self.emb]  # filtering the tokens - whether they're in word2vec
        if not valid_tokens:
            return None
        vecs = [self.emb[token] for token in valid_tokens]  # calculating the vectors and putting them into an array
        return np.mean(vecs, 0)  # return the mean vector


    def get_vectors(self):
        """
        Calculate the vector of the preprocessed text 'pp_text' in each
        inner dict of data_dict. Add a new key 'vector'
        to each inner dict, where the value is the mean of individual word vectors
        as returned by get_vector().

        If 'pp_text' is an empty list, or none of the words in 'pp_text' are
        in word2vec, the value of 'vector' is None.

        :param data_dict: a nested dictionary where inner dicts have key 'pp_text'
        """
        for inner_dict in self.data.values():  # iteration through inner dicts
            if not inner_dict['pp_text']:  # if the pp_text key has no value
                inner_dict['vector'] = None  # creating valueless entry for vector
            else:  # if there is a value for pp_text
                inner_dict['vector'] = self.get_vector(inner_dict['pp_text'])  # computing the vector


    def k_most_similar(self, query, k=5):
        """
        Find the k most similar entries in data_dict to the query.

        The query is first preprocessed, then a mean word vector is calculated for it.

        Return a list of tuples of length k where each tuple contains
        the id of the data_dict entry and the cosine similarity score between the
        data_dict entry and the user query.

        In some cases, the similarity cannot be calculated. For example:
        - If none of the preprocessed token strings are in word2vec for an entry in data_dict.
        If you built data_dict according to the instructions, the value of 'vector'
        is None in these cases, and those entries should simply not be considered.
        - If a vector for the query can't be calculated, return an empty list.

        :param query: a query string as typed by the user
        :param k: number of top results to return
        :return: a list of tuples of length k, each containing an id and a similarity score,
        or an empty list if the query can't be processed
        """
        vec = self.get_vector(self.preprocess_text(query))  # preprocessing the query and computing the vector
        if vec is None:
            return []  # return empty list if there is no vector
        similarities = []
        for key, inner_dict in self.data.items():  # iteration through dictionary items
            if inner_dict['vector'] is not None:
                similarity = cosine_similarity(vec, inner_dict['vector'])  # computing the similarity if vector isn't empty
                similarities.append((key, similarity))  # adding the similarity to the list
        similarities.sort(key=lambda x: x[1], reverse=True)  # sort the similarities from the biggest to the smallest
        return similarities[:k]


    def recommender_app(self):
        """
        Implement your recommendation system here.

        - Repeatedly prompt the user to type a query
            - Print the description and the url of the top 5 most similar,
            or "No Results Found" if appropriate
            - Return when the query is "quit" (without quotes)

        :param data_dict: nested dictionaries containing
        description,url,tokens,and vectors for each description
        in the input data
        """
        while True:  # repeatedly prompt the user
            prompt = f"\nType a query. To quit type 'quit' (case not important; without the quotation marks). "
            query = input(prompt)
            if query.lower() == "quit":  # instruction breaking the loop
                break
            else:
                most_similar = self.k_most_similar(query, 5)  # finding the most similar to the query
                if not most_similar:
                    print("No results found")
                    # webbrowser.open("https://youtube.com/shorts/RwIhNXAXgQ8?si=Srhe4SFbDQBeKWnv", 0, True)
                else:
                    print("The 5 most similar talks: ")  # print
                    for i, (id_number, similarity) in enumerate(most_similar, start=1):  # printing 5 most similar talks
                        entry = self.data[id_number]
                        print(f"{i}. Description: {entry['description']}")
                        print(f"URL: {entry['url']}")


def parseargs():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--spacy_model", required=True, help="Path to the spaCy model")
    ap.add_argument("-e", "--embeddings", required=True, help="Path to the embeddings file")
    ap.add_argument("-d", "--data", required=True, help="path to input data file")
    return ap.parse_args()


def main(args):
    """
    Bring it all together here.
    """
    app = RecommenderApp(args.spacy_model, args.embeddings, args.data)
    app.recommender_app()


if __name__ == '__main__':
    main(parseargs())
