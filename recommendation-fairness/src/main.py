import time
import pandas as pd
import random
import os
from df_manager import DataFrameManager
from recomm_techniques.user_based_collaborative_filtering import UserBasedCollaborativeFiltering


class RecommendationAssignment:
    def __init__(self):
        self.df_manager = DataFrameManager()
        self.user_based_collaborative_filtering = UserBasedCollaborativeFiltering(self.df_manager)
        
        self.first_assignment()
        

    def first_assignment(self):
        """
        Calculates and shows what is required for the completion
        of the first assignment
        """
        print("#################  FIRST ASSIGNMENT  #################\n")
        
        #point a - Shows the few rows of every table and count of elements
        self.df_manager.show_dataset()
        
        # point b - Implemented the Pearson Correlation function and example
        #           to show some results
        self.example_similarity(45, 89)
        
        # point c & d - Implemented the prediction function for a user-based collaborative filtering approach
        #               Example below as the assignment requests 
        self.example_user_based_collaborative_filtering()
        
        print("\n#####################################################")

    
    def example_similarity(self, userA: int, userB: int):
        similarity = self.user_based_collaborative_filtering.pearson_correlation(userA, userB)
        print("\n\n_______________________EXAMPLE PEARSON SIMILARITY BETWEEN TWO USERS_____________________________")
        print("\nThe similarity between user:" + str(userA) + " and user:" + str(userB) + " (using Pearson similarity) is: " + str(similarity))
        print("________________________________________________________________________________________________\n")
        
    
    def example_user_based_collaborative_filtering(self):
        # Shows for a random user, the 10 most similar users
        randomId = random.randint(1, self.df_manager.get_users_count())
        self.user_based_collaborative_filtering.show_top_x_similar_users(randomId, 10)

        # Shows the top 10 predictions for the same user
        print("\n\n(The prediction function implementation takes in account all the\n" + 
              " similarities between the user we want to suggest movies to and \n" +
              " all the other users - so it might take a few seconds to output )")   
        start_time = time.time()
        self.user_based_collaborative_filtering.show_top_x_recommendations(randomId, 10)
        end_time = time.time()

        elapsed_time = end_time - start_time

        print("Time taken to calculate predictions:", elapsed_time, "seconds")
                


if __name__ == "__main__":
    RecommendationAssignment()
