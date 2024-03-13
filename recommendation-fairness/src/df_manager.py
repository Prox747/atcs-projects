import os
import pandas as pd
from pandasql import sqldf

class DataFrameManager:
    def __init__(self):
        self.links_df = None
        self.movies_df = None
        self.ratings_df = None
        self.tags_df = None
    
    def initialize_dataframes(self):
        self.links_df = self.create_data_frame('links.csv')
        self.movies_df = self.create_data_frame('movies.csv')
        self.ratings_df = self.create_data_frame('ratings.csv')    
        self.tags_df = self.create_data_frame('tags.csv')
    
    def create_data_frame(self, file_name: str):
        project_folder = os.path.dirname(os.path.dirname(__file__))
        file_path = os.path.join(project_folder, 'data', file_name)
        return pd.read_csv(file_path)
    
    def show_dataset(self):
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
        
    
    # Gets the total number of users
    def get_users_count(self):
         return len(set(self.ratings_df['userId']))
    

    def calc_user_ratings_mean(self, userId: int):
        """
        Calculates the mean of ratings for a particular user
        """
        grouped_by_user = self.ratings_df.groupby('userId')
        return grouped_by_user.get_group(userId).rating.mean()

    def calc_users_ratings_mean(self):
        """
        Calculates the mean of ratings for each user
        """
        grouped_by_user = self.ratings_df.groupby('userId')
        return grouped_by_user.rating.mean()

    def get_user_ratings_df(self, userId: int):
        """
        Obtains a particular user's ratings dataframe.
        
        Returns:
            DataFrame: A dataframe userId,movieId,rating 
        """
        grouped_by_user = self.ratings_df.groupby('userId')
        return grouped_by_user.get_group(userId)

    def get_common_movies_rated_by_users_as_map(self, userA: int, userB: int):
        """
        Given two users, returns a map {MOVIEID, (RatingUserA, RatingUserB)} for every movie
        that both users have rated.
        
        This function is useful for calculating Pearson correlation (calculates similarity between two users).
        
        Args:
            userA (int): The ID of the first user.
            userB (int): The ID of the second user.
            
        Returns:
            dict: A dictionary mapping movie IDs to tuples of ratings for userA and userB.
        """
        userdfA = self.get_user_ratings_df(userA)
        userdfB = self.get_user_ratings_df(userB)
        
        # Define the SQL query to get movie ratings for the two users
        query = f"""
        SELECT a.movieId, a.rating AS rating_userA, b.rating AS rating_userB
        FROM userdfA AS a
        INNER JOIN userdfB AS b ON a.movieId = b.movieId
        WHERE a.userId = {userA} AND b.userId = {userB}
        """
        
        # Execute the query using pandasql
        result_df = sqldf(query, locals())
        
        # Convert the result DataFrame to a dictionary
        movie_ratings_map = {}
        for index, row in result_df.iterrows():
            movie_ratings_map[row['movieId']] = (row['rating_userA'], row['rating_userB'])
        
        return movie_ratings_map
