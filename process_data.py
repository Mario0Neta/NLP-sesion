import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from sklearn.model_selection import train_test_split
import nltk

nltk.download("vader_lexicon")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download('averaged_perceptron_tagger_eng')


class ProcessData:
    df: pd.DataFrame
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.DataFrame

    def __init__(self, df: pd.DataFrame, independent: str, dependent: str):
        self.df = df
        self.independent = independent
        self.dependent = dependent
        self.stopwords_en = stopwords.words("english")
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = TweetTokenizer()

    def data_cleaning(self) -> pd.DataFrame:
        stopwords_rem = False
        reconstructed_list = []
        for each_text in self.df[self.independent]:
            lemmatized_tokens = self.__lemmatize_reviews(each_text, stopwords_rem)
            reconstructed_list.append(" ".join(lemmatized_tokens))
        self.df[f'cleaned_{self.independent}'] = reconstructed_list
        return self.df[[f'cleaned_{self.independent}', self.dependent]]

    def data_split(self) -> tuple:
        X = self.df[[f'cleaned_{self.independent}']]
        y = self.df[self.dependent]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        return X_train, X_test, y_train, y_test

    def __lemmatize_reviews(self, each_text, stopwords_rem) -> list:
        lemmatized_tokens = []
        tokens = self.tokenizer.tokenize(each_text.lower())
        pos_tags = pos_tag(tokens)
        for each_token, tag in pos_tags:
            if tag.startswith("NN"):
                pos = "n"
            elif tag.startswith("VB"):
                pos = "v"
            else:
                pos = "a"
            lemmatized_token = self.lemmatizer.lemmatize(each_token, pos)
            if stopwords_rem:  # False
                if lemmatized_token not in self.stopwords_en:
                    lemmatized_tokens.append(lemmatized_token)
            else:
                lemmatized_tokens.append(lemmatized_token)
        return lemmatized_tokens