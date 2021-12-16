import tez
import numpy
import pandas
import torch
import pandas
from sklearn import model_selection, metrics, preprocessing

class MovieDataset:
    def __init__(self, users, movies, ratings) -> None:
        self.users = users
        self.movies = movies
        self.ratings = ratings
    
    def __len__(self):
        return len(self.users)

    def __getitem__(self, item):
        user = self.users[item]
        movie = self.movies[item]
        rating = self.ratings[item]
        return {
            "users": torch.tensor(user, dtype=torch.long), 
            "movies": torch.tensor(movie, dtype=torch.long), 
            "ratings": torch.tensor(rating, dtype=torch.float)
            }

class RecSysModel(tez.Model):
    def __init__(self, num_users, num_movies):
        super().__init__()
        self.user_embed = torch.nn.Embedding(num_users, 32)
        self.movie_embed = torch.nn.Embedding(num_movies, 32)
        self.out = torch.nn.Linear(64, 1)
        self.step_scheduler_after = 'epoch'

    def monitor_metrics(self, target, rating):
        target = target.detach().cpu().numpy()
        rating = rating.detach().cpu().numpy()
        return {
            'RMSE': numpy.sqrt(metrics.mean_squared_error(rating, target))
        }

    def fetch_optimizer(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        return opt

    def fetch_scheduler(self):
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.7)
        return scheduler

    def forward(self, users, movies, ratings=None): 
        user_embeds = self.user_embed(users)
        movie_embeds = self.movie_embed(movies)
        output = torch.cat([user_embeds, movie_embeds], dim=1)
        output = self.out(output)
        # if ratings: 
        loss = torch.nn.MSELoss()(output, ratings.view(-1, 1))
        calc_metrics = self.monitor_metrics(output, ratings.view(-1, 1))
        return output, loss, calc_metrics
        # else: 
            # return output

def train(): 
    df = pandas.read_csv("./ml-latest-small/ratings.csv")

    labelUser = preprocessing.LabelEncoder()
    labelMovies = preprocessing.LabelEncoder()

    df.userId = labelUser.fit_transform(df.userId.values)
    df.movieId = labelMovies.fit_transform(df.movieId.values)

    df_train, df_valid = model_selection.train_test_split(df, test_size=0.1, random_state=42, stratify=df.rating.values)

    train_dataset = MovieDataset(users=df_train.userId.values, movies=df_train.movieId.values, ratings=df_train.rating.values)
    valid_dataset = MovieDataset(users=df_valid.userId.values, movies=df_valid.movieId.values, ratings=df_valid.rating.values)

    # print(len(labelMovies.classes_))
    model = RecSysModel(num_users=len(labelUser.classes_), num_movies=len(labelMovies.classes_))
    model.fit(
        train_dataset, valid_dataset, train_bs=1024, valid_bs=1024, fp16=True
    )


if __name__ == '__main__':
    train()