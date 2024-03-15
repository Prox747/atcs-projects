import os
import time
import pandas as pd

class DataSetManager:
    def __init__(self):
        self.initialize_dataframes()
        self.ratings_grouped_by_user = self.ratings_df.groupby('userId')
        self.user2ratings = self.create_users_ratings_map()
        self.per_movie_variance = self.calc_per_movie_variance()

    
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
    
    # Gets a movie by its ID
    def get_movie(self, movieId: int):
        return self.movies_df[self.movies_df['movieId'] == movieId]
    

    # Gets all movies that the user has not rated
    def get_movies_not_rated_by_user(self, userId):
        user_movies = set(self.get_user_ratings_df(userId)['movieId'])
        all_movies = set(self.movies_df['movieId'])
        return list(all_movies - user_movies)

    
    # Gets the ratings for a particular user
    def calc_user_ratings_mean(self, userId: int):
        """
        Calculates the mean of ratings for a particular user
        """
        return self.ratings_grouped_by_user.get_group(userId).rating.mean()


    # Gets the mean of ratings for each user
    def get_users_ratings_mean(self):
        """
        Calculates the mean of ratings for each user
        """
        return self.ratings_grouped_by_user.rating.mean()


    # Returns a map UserId2(MovieId2Ratings)
    def create_users_ratings_map(self):
        user_ratings = {}
        for userId, user_df in self.ratings_grouped_by_user:
            # Convert DataFrame to dictionary with movieId as key and rating as value
            movie_rating_map = dict(zip(user_df['movieId'], user_df['rating']))
            # Store the mapping for the user
            user_ratings[userId] = movie_rating_map
        return user_ratings
    
    
    def get_users_ratings_map(self):
        return self.user2ratings


    # Gets the ratings dataframe for a particular user
    def get_user_ratings_df(self, userId: int):
        """
        Obtains a particular user's ratings dataframe.
        
        Returns:
            DataFrame: A dataframe userId,movieId,rating 
        """
        return self.ratings_grouped_by_user.get_group(userId)
    
    
    # Gets the ratings for a particular user
    def get_user_ratings_map(self, userId: int):
        """
        Obtains a particular user's ratings map.
        
        Returns:
            Map: A map userId2(movieId2rating)
        """
        return self.user2ratings.get(userId)


    # Calculates the variance of ratings for each movie
    def calc_per_movie_variance(self):
        """
        Calculates the variance of ratings for each movie.
        
        Returns:
            Series: A series of movieId2variance
        """
        return self.ratings_df.groupby('movieId').rating.var()


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
        # Get the rating dictionaries for the two users
        ratings_userA = self.user2ratings.get(userA, {})
        ratings_userB = self.user2ratings.get(userB, {})
        
        # Find common movies between the two users
        common_movies = set(ratings_userA.keys()) & set(ratings_userB.keys())
        
        # Create a map of common movie ratings for userA and userB
        common_movies_ratings_map = {}
        for movieId in common_movies:
            rating_userA = ratings_userA.get(movieId)
            rating_userB = ratings_userB.get(movieId)
            common_movies_ratings_map[movieId] = (rating_userA, rating_userB)
        
        return common_movies_ratings_map
