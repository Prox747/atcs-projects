import math
import random
import time
import pandas as pd

from numpy import sort
from df_manager import DataFrameManager

class UserBasedCollaborativeFiltering:
    def __init__(self, df_manager: DataFrameManager):
        self.df_manager = df_manager
    
    
    # Calculates Pearson Correlation between two users
    def pearson_correlation(self, userA: int, userB: int):
        # Creates a map containing all common movies between the users and their ratings for each movie
        common_movies_with_ratings = self.df_manager.get_common_movies_rated_by_users_as_map(userA, userB)
        
        # will cause division by zero
        # no movies in common, then we cannot
        # evaluate similarity -> 0
        if common_movies_with_ratings == {}:
            return 0

        userAMean = self.df_manager.calc_user_ratings_mean(userA)
        userBMean = self.df_manager.calc_user_ratings_mean(userB)

        numerator = 0
        sum_of_squared_differences_B = 0 
        sum_of_squared_differences_A = 0

        # {MOVIEID, (RatingUserA, RatingUserB)}
        for ratings in common_movies_with_ratings.values():
            numerator += (ratings[0] - userAMean)*(ratings[1] - userBMean)
            sum_of_squared_differences_A += (ratings[0] - userAMean) ** 2
            sum_of_squared_differences_B += (ratings[1] - userBMean) ** 2
            
        denominator = (math.sqrt(sum_of_squared_differences_A)) * (math.sqrt(sum_of_squared_differences_B))

        if denominator == 0:
            return 0
        
        similarity = numerator / denominator
        
        # At times, because of float imprecision, it becomes
        # sim = 1.000000002 or -1.000000002, we want it rounded
        if similarity > 1 or similarity < -1:
            math.ceil(similarity)

        return similarity
    
    
    # Calculates the top 'num_elements' similar users to the given userId
    def get_top_x_similar_users(self, userId: int, num_elements: int):
        similar_users = {}
        
        for otherId in range(1, self.df_manager.get_users_count()):
            if otherId != userId:
                similar_users[otherId] = self.pearson_correlation(userId, otherId)
        
        # Sort the dictionary by values in descending order
        # key=lambda x: x[1] -> sort by the value
        sorted_similar_users = sorted(similar_users.items(), key=lambda x: x[1], reverse=True)
        
        # Take the first 'num_elements'
        return sorted_similar_users[:num_elements]


    # Shows the top 'num_elements' similar users to the given userId
    def show_top_x_similar_users(self, userId: int, num_elements: int):
        top_similar_users = self.get_top_x_similar_users(userId, num_elements)
        
        result_df = pd.DataFrame(top_similar_users, columns=['UserId', 'Similarity'])
        
        print(f"The top {num_elements} similar users to user {userId} are:")
        print(result_df.to_string(index=False))
        
    

    # Calculates the top predicted ratings for the given userId
    def get_top_x_recommendations(self, userId: int, num_elements: int):
        # Get all movies that the user has not rated
        # start = time.time()
        movies_not_rated = self.df_manager.get_movies_not_rated_by_user(userId)
        # end = time.time()
        # print(f"\nTime to calc movies not rated by user set: {end - start}")
        
        # Get the top 30 similar users to the user
        most_similar_users = self.get_top_x_similar_users(userId, 30)
        
        all_users_ratings_mean = self.df_manager.get_users_ratings_mean()
        
        # Calculate the predicted rating for each movie
        predicted_ratings = []
        for movieId in movies_not_rated:
            rating = self.predict_rating(userId, most_similar_users, all_users_ratings_mean, movieId)
            predicted_ratings.append((movieId, rating))
        
        # Sort the dictionary by values in descending order
        # key=lambda x: x[1] -> sort by the value
        predicted_ratings.sort(key=lambda x: x[1], reverse=True)
        
        # Take the first 'num_elements'
        return predicted_ratings[:num_elements]


    def predict_rating(self, userId: int, similar_users: list[tuple], all_users_ratings_mean: pd.Series, movieId: int): 
        # Calculate the predicted rating for the given movie
        numerator = 0
        denominator = 0
        
        for otherId, similarity in similar_users:
            # Get the rating of the user for the movie (we actually get a series of rating of length 1)
            user_ratings = self.df_manager.get_user_ratings_df(otherId)
            rating = user_ratings[user_ratings['movieId'] == movieId]['rating']
            # For now we skip the user if it has not rated the movie
            if len(rating) > 0:
                # The user has rated the movie, we take the rating
                rating = rating.values[0]
                mean = all_users_ratings_mean[otherId]
                numerator += similarity * (rating - mean)
                denominator += similarity
        
        if denominator == 0:
            return 0
        
        predicted_rating = all_users_ratings_mean[userId] + (numerator / denominator)
        return predicted_rating
        

    # Shows the top 'num_elements' recommendations for the given userId
    def show_top_x_recommendations(self, userId: int, num_elements: int):
        # Get the top 'num_elements' recommendations for the given userId
        recommendations = self.get_top_x_recommendations(userId, num_elements)
        
        result_df = pd.DataFrame(recommendations, columns=['Movie Title', 'Predicted Rating'])

        print(f"\nThe top {num_elements} recommendations for user {userId} are:")
        print(result_df.to_string(index=False))
        
            
        