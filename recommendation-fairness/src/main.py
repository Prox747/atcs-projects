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
        
        # inizialize dataframes
        self.df_manager.initialize_dataframes()
        
        #point a - Shows the few rows of every table
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
                


if __name__ == "__main__":
    RecommendationAssignment()
