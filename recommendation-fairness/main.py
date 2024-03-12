import pandas as pd
import random as rand
import utils
import os


class RecommendationAssignment:

    def __init__(self):
        self.project_folder = os.path.dirname(__file__)

        self.create_data_frames()
        self.print_df_heads()

        self.users_ratings_mean = self.calc_users_ratings_mean()
        self.user_movie_ratings = self.create_user_ratings_groups()

        print(utils.pearson_correlation(self.user_movie_ratings, self.users_ratings_mean, 45, 89))



    
    def create_data_frames(self):
        # Costruisci i percorsi completi dei file CSV
        links_file_path = os.path.join(self.project_folder, 'ml-latest-small', 'links.csv')
        movies_file_path = os.path.join(self.project_folder, 'ml-latest-small', 'movies.csv')
        ratings_file_path = os.path.join(self.project_folder, 'ml-latest-small', 'ratings.csv')
        tags_file_path = os.path.join(self.project_folder, 'ml-latest-small', 'tags.csv')

        # Leggi i file CSV
        self.links_df = pd.read_csv(links_file_path)
        self.movies_df = pd.read_csv(movies_file_path)
        self.ratings_df = pd.read_csv(ratings_file_path)
        self.tags_df = pd.read_csv(tags_file_path)

    # Prints the first few rows of each table
    def print_df_heads(self):
        print("\n-------- LINKS.CSV --------")
        print("Number of elements in links.csv:", len(self.links_df))
        print("First few rows of links.csv:")
        print(self.links_df.head())


        print("\n-------- MOVIES.CSV --------")
        print("Number of elements in movies.csv:", len(self.movies_df))
        print("First few rows of movies.csv:")
        print(self.movies_df.head())

        print("\n-------- RATINGS.CSV --------")
        print("Number of elements in ratings.csv:", len(self.ratings_df))
        print("First few rows of ratings.csv:")
        print(self.ratings_df.head())

        print("\n-------- TAGS.CSV --------")
        print("Number of elements in tags.csv:", len(self.tags_df))
        print("First few rows of tags.csv:")
        print(self.tags_df.head())
    

    # Calculates the mean of ratings for each users
    def calc_users_ratings_mean(self):
        grouped_by_user = self.ratings_df.groupby('userId')

        return grouped_by_user.rating.mean()


    def create_user_ratings_groups(self):
        # Groups the DataFrame by user ID to obtain ratings for each user
        grouped_by_user = self.ratings_df.groupby('userId')

        # Creates a dictionary where the keys are movie IDs and the values are tuples containing ratings
        user_ratings = {}
        for user_id, user_df in grouped_by_user:
            ratings = dict(zip(user_df['movieId'], user_df['rating']))
            user_ratings[user_id] = ratings
        
        return user_ratings


if __name__ == "__main__":
    RecommendationAssignment()
