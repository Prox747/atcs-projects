import math
import time
import pandas as pd
import numpy as np
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
            rating_diff_A = ratings[0] - userAMean
            rating_diff_B = ratings[1] - userBMean
            numerator += rating_diff_A * rating_diff_B
            sum_of_squared_differences_A += rating_diff_A ** 2
            sum_of_squared_differences_B += rating_diff_B ** 2
            
        denominator = (math.sqrt(sum_of_squared_differences_A)) * (math.sqrt(sum_of_squared_differences_B))

        if denominator == 0:
            return 0
        
        similarity = numerator / denominator
        
        # At times, because of float imprecision, it becomes
        # sim = 1.000000002 or -1.000000002, we want it rounded
        if similarity > 1 or similarity < -1:
            math.ceil(similarity)

        return similarity
    

    def pearson_correlation_weighted(self, userA: int, userB: int):
        """
        Calculates the Pearson Correlation between two users, but with variance weights
        
        The variance weights are calculated as the inverse variance of the ratings for a movie
        
        """
        # Creates a map containing all common movies between the users and their ratings for each movie
        common_movies_with_ratings = self.ds_manager.get_common_movies_rated_by_users_as_map(userA, userB)
        
        movie2variance_normalized = self.ds_manager.movie2norm_var
        
        # If no common movies, return 0
        if not common_movies_with_ratings:
            return 0

        userAMean = self.ds_manager.calc_user_ratings_mean(userA)
        userBMean = self.ds_manager.calc_user_ratings_mean(userB)

        numerator = 0
        sum_of_squared_differences_B = 0 
        sum_of_squared_differences_A = 0

        # Calculate Pearson correlation
        for movie, (ratingA, ratingB) in common_movies_with_ratings.items():
            # Calculate differences from mean ratings
            rating_diff_A = ratingA - userAMean
            rating_diff_B = ratingB - userBMean
            # Calculate weight for the current movie based on its variance
            weight = movie2variance_normalized[movie]
            
            numerator += rating_diff_A * rating_diff_B * weight
            sum_of_squared_differences_A += (rating_diff_A ** 2) * weight
            sum_of_squared_differences_B += (rating_diff_B ** 2) * weight
                

        denominator = math.sqrt(sum_of_squared_differences_A) * math.sqrt(sum_of_squared_differences_B)

        if denominator == 0:
            return 0
        
        similarity = numerator / denominator
        
        # Ensure similarity is within [-1, 1]
        #similarity = max(-1, min(similarity, 1))

        return similarity

    
    def get_all_sim_for_user(self, userId: int):
        """
        Calculates and returns all the similarity values between
        userId and all the other users
        """
        return self.get_top_x_similar_users(userId, self.ds_manager.get_users_count())
    
    
    # Calculates the top 'num_elements' similar users to the given userId
    def get_top_x_similar_users(self, userId: int, num_elements: int, sim_method: str = "pcc"):
        similar_users = {}
        
        sim_func = self.pearson_correlation if sim_method == "pcc" else self.pearson_correlation_weighted
        
        for otherId in range(1, self.ds_manager.get_users_count()):
            if otherId != userId:
                similar_users[otherId] = sim_func(userId, otherId)
        
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
    def get_top_x_recommendations(self, userId: int, num_elements: int, neighbourhood_size: int = -1, sim_method: str = "pcc"):
        # Get all movies that the user has not rated
        movies_not_rated = self.ds_manager.get_movies_not_rated_by_user(userId)
        
        # Get the top 'neighbourhood_size' similar users to userId (40 should be enough)
        if neighbourhood_size == -1:
            neighbourhood_size = self.ds_manager.get_users_count()
        neighbourhood_similarities = self.get_top_x_similar_users(userId, neighbourhood_size, sim_method)
        
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
    def show_top_x_recommendations(self, userId: int, num_elements: int, neighbourhood_size: int = -1, sim_method: str = "pcc"):
        # Get the top 'num_elements' recommendations for the given userId
        recommendations = self.get_top_x_recommendations(userId, num_elements, neighbourhood_size, sim_method)
        
        # Fetch movie titles from movies_df
        movie_titles = self.ds_manager.movies_df.set_index('movieId')['title']
        
        # Replace movie IDs with movie titles
        recommendations_with_titles = [(movie_titles[movieId], rating) for movieId, rating in recommendations]
        
        # Create DataFrame from recommendations with movie titles
        result_df = pd.DataFrame(recommendations_with_titles, columns=['Movie Title', 'Predicted Rating'])

        print(f"\nThe top {num_elements} recommendations for user {userId}")
        print(f"(for a neighbourhood of {neighbourhood_size if neighbourhood_size == -1 else 'all the'} users):")
        print(result_df.to_string(index=False))
        
            
        