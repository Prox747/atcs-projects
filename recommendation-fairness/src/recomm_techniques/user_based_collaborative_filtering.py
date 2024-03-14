import math
import random
import time
import pandas as pd

from numpy import sort
from ds_manager import DataSetManager

class UserBasedCollaborativeFiltering:
    def __init__(self, ds_manager: DataSetManager):
        self.ds_manager = ds_manager
    
    
    # Calculates Pearson Correlation between two users
    def pearson_correlation(self, userA: int, userB: int):
        # Creates a map containing all common movies between the users and their ratings for each movie
        common_movies_with_ratings = self.ds_manager.get_common_movies_rated_by_users_as_map(userA, userB)
        
        # will cause division by zero
        # no movies in common, then we cannot
        # evaluate similarity -> 0
        if common_movies_with_ratings == {}:
            return 0

        userAMean = self.ds_manager.calc_user_ratings_mean(userA)
        userBMean = self.ds_manager.calc_user_ratings_mean(userB)

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
    
    
    def get_all_sim_for_user(self, userId: int):
        """
        Calculates and returns all the similarity values between
        userId and all the other users
        """
        return self.get_top_x_similar_users(userId, self.ds_manager.get_users_count())
    
    
    # Calculates the top 'num_elements' similar users to the given userId
    def get_top_x_similar_users(self, userId: int, num_elements: int):
        similar_users = {}
        
        for otherId in range(1, self.ds_manager.get_users_count()):
            if otherId != userId:
                similar_users[otherId] = self.pearson_correlation(userId, otherId)
        
        # Sort the dictionary by values in descending order
        # key=lambda x: x[1] -> sort by the value
        sorted_similar_users = sorted(similar_users.items(), key=lambda x: x[1], reverse=True)
        
        # Take the first 'num_elements' if requested, else give all the similarities
        return sorted_similar_users[:num_elements]
    

    # Shows the top 'num_elements' similar users to the given userId
    def show_top_x_similar_users(self, userId: int, num_elements: int):
        top_similar_users = self.get_top_x_similar_users(userId, num_elements)
        
        result_df = pd.DataFrame(top_similar_users, columns=['UserId', 'Similarity'])
        
        print(f"The top {num_elements} similar users to user {userId} are:")
        print(result_df.to_string(index=False))
        
    
    # Calculates the top predicted ratings for the given userId
    def get_top_x_recommendations(self, userId: int, num_elements: int, neighbourhood_size: int = -1):
        # Get all movies that the user has not rated
        movies_not_rated = self.ds_manager.get_movies_not_rated_by_user(userId)
        
        # Get the top 'neighbourhood_size' similar users to userId (40 should be enough)
        if neighbourhood_size is -1:
            neighbourhood_size = self.ds_manager.get_users_count()
        neighbourhood_similarities = self.get_top_x_similar_users(userId, neighbourhood_size)
        
        # Calculate the predicted rating for each movie
        predicted_ratings = []
        for movieId in movies_not_rated:
            rating = self.predict_rating(userId, neighbourhood_similarities, movieId)
            predicted_ratings.append((movieId, rating))
        
        # Sort the dictionary by values in descending order
        # key=lambda x: x[1] -> sort by the value
        predicted_ratings.sort(key=lambda x: x[1], reverse=True)
        
        # Take the first 'num_elements'
        return predicted_ratings[:num_elements]


    def predict_rating(self, userId: int, similar_users: list[tuple], movieId: int, steps: int = 0): 
        # Calculate the predicted rating for the given movie
        numerator = 0
        denominator = 0
        
        for otherId, similarity in similar_users:
            # Get the rating of the user for the movie
            user_ratings = self.ds_manager.get_users_ratings_map().get(otherId)
            rating = user_ratings.get(movieId)
            # For now we skip the user if it has not rated the movie
            if rating is not None:
                mean = self.ds_manager.calc_user_ratings_mean(otherId)
                numerator += similarity * (rating - mean)
                denominator += similarity
        
        if denominator == 0:
            return 0
        
        predicted_rating = self.ds_manager.calc_user_ratings_mean(userId) + (numerator / denominator)
        return predicted_rating
        

    # Shows the top 'num_elements' recommendations for the given userId
    def show_top_x_recommendations(self, userId: int, num_elements: int, neighbourhood_size: int = -1):
        # Get the top 'num_elements' recommendations for the given userId
        recommendations = self.get_top_x_recommendations(userId, num_elements, neighbourhood_size)
        
        # Fetch movie titles from movies_df
        movie_titles = self.ds_manager.movies_df.set_index('movieId')['title']
        
        # Replace movie IDs with movie titles
        recommendations_with_titles = [(movie_titles[movieId], rating) for movieId, rating in recommendations]
        
        # Create DataFrame from recommendations with movie titles
        result_df = pd.DataFrame(recommendations_with_titles, columns=['Movie Title', 'Predicted Rating'])

        print(f"\nThe top {num_elements} recommendations for user {userId}")
        print(f"(for a neighbourhood of {neighbourhood_size if neighbourhood_size is not -1 else 'all the'} users):")
        print(result_df.to_string(index=False))
        
            
        