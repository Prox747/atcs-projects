import math
from df_manager import DataFrameManager

class SimCalc:
    def __init__(self, df_manager: DataFrameManager):
        self.df_manager = df_manager
         
    # Calculates Pearson Correlation
    def pearson_correlation(self, userA: int, userB: int):
        # Creates a map containing all common movies between the users and their ratings for each movie
        common_movies_with_ratings = self.df_manager.get_common_movies_rated_by_users_as_map(userA, userB)

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

        similarity = numerator / denominator

        return similarity