import math
import random

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

        return similarity
    
    
    # Calculates the top 'num_elements' similar users to the given userId
    def show_top_x_similar_users(self, userId: int, num_elements: int):
        similar_users = {}
        
        for otherId in range(1, self.df_manager.get_users_count()):
            if otherId != userId:
                similar_users[otherId] = self.pearson_correlation(userId, otherId)
        
        # Sort the dictionary by values in descending order
        sorted_similar_users = sorted(similar_users.items(), key=lambda x: x[1], reverse=True)
        
        # Take the first 'num_elements'
        top_similar_users = sorted_similar_users[:num_elements]
        
        print(f"I {num_elements} utenti più simili all'utente: {userId} sono:\n")
        for user, value in top_similar_users:
            print(f"Utente {user}, sim_score: {value}")
        
            
        