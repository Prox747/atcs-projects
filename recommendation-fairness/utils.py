import random
import pandas as pd
import math

# Calculates Pearson Correlation
def pearson_correlation(user_ratings: dict, users_ratings_mean: pd.Series, userA: int, userB: int):
    # Creates a map containing all common movies between the users and their ratings for each movie
    common_movies_with_ratings = create_movie_ratings_map(user_ratings, userA, userB)
    userAMean = users_ratings_mean.get(userA)
    userBMean = users_ratings_mean.get(userB)

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


# Given two users, it creates a map {MOVIEID, (RatingUserA, RatingUserB)}
# Useful for calculating Pearson correlation (Calculates similarity between two users)
def create_movie_ratings_map(user_ratings: dict, userA: int, userB: int):
    # Create a set containing movies rated by both users
    user_a_ratings = user_ratings.get(userA, {})
    user_b_ratings = user_ratings.get(userB, {})
    common_movies = set(user_a_ratings.keys()) & set(user_b_ratings.keys())

    # Create the map {MOVIEID, (RatingUserA, RatingUserB)}
    movie_ratings_map = {}
    for movie_id in common_movies:
        movie_ratings_map[movie_id] = (user_a_ratings.get(movie_id), user_b_ratings.get(movie_id))

    return movie_ratings_map
