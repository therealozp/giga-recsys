import torch 
import pandas
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from pandas import isnull, notnull

movies_df = "C:/Users/Admin/code/queb-recommendation-system/ml-latest-small/movies.csv"

def getDataframeMovies(file): 
    movieColumns = ['movieId', 'title', 'genres']
    movies = pandas.read_csv(file, sep=",", names=movieColumns)
    return movies

def tf_idfMatrix(movies):
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0)
    tfidfmatix = tf.fit_transform(movies['genres'])
    return tfidfmatix

def cosineSimlarity(matrix): 
    cosSim = linear_kernel(matrix, matrix)
    return cosSim

class Model(object): 
    def __init__(self, movieDataset):
        self.movies = getDataframeMovies(movieDataset)
        self.matrix = None
        self.cosineSimilarity = None
    
    def buildModel(self): 
        print(self.movies)
        self.movies['genres'] = self.movies['genres'].str().split("|")
        # self.movies['genres'] = self.movies['genres'].fillna("").astype('str')
        self.matrix = tf_idfMatrix(self.movies)
        self.cosineSimilarity = cosineSimlarity(self.matrix)
        
    def refresher(self): 
        self.buildModel()

    def fit(self): 
        self.refresher()

    def genreRecommend(self, title, topX): 
        titles = self.movies['title']
        # print(titles[1])
        indicies = pandas.Series(self.movies.index, index=self.movies['title'])
        # print(indicies[1:12])
        idx = indicies[title]

        simiScores = list(enumerate(self.cosineSimilarity[idx]))

        simiScores = sorted(simiScores, key=lambda x: x[1], reverse=True)
        simiScores = simiScores[1:topX + 1]
        movieIndicies = [i[0] for i in simiScores]
        return simiScores, titles.iloc[movieIndicies].values


movieDs = Model(movieDataset=movies_df)
print(movieDs.genreRecommend(title='GoldenEye (1995)', topX=5))
# print(getDataframeMovies(movies_df))