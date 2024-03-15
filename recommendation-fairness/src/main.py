import time
import pandas as pd
import random
import os
from ds_manager import DataSetManager
from recomm_techniques.user_based_collaborative_filtering import UserBasedCollaborativeFiltering


class RecommendationAssignment:
    def __init__(self):
        self.ds_manager = DataSetManager()
        self.user_based_collaborative_filtering = UserBasedCollaborativeFiltering(self.ds_manager)
        
        self.first_assignment()
        

    def first_assignment(self):
        """
        Calculates and shows what is required for the completion
        of the first assignment
        """
        print("#################  FIRST ASSIGNMENT  #################\n")
        
        #point a - Shows the few rows of every table and count of elements
        self.ds_manager.show_dataset()
        
        # point b - Implemented the Pearson Correlation function and example
        #           to show some results
        self.example_similarity(45, 89)
        
        # point c & d - Implemented the prediction function for a user-based collaborative filtering approach
        #               Example below as the assignment requests 
        self.example_user_based_collaborative_filtering()
        
        print("\n#####################################################")

    
    def example_similarity(self, userA: int, userB: int):
        similarity_w = self.user_based_collaborative_filtering.pearson_correlation_weighted(userA, userB)
        similarity = self.user_based_collaborative_filtering.pearson_correlation(userA, userB)
        print("\n\n_______________________EXAMPLE PEARSON SIMILARITY BETWEEN TWO USERS_____________________________")
        print("\nThe similarity between user:" + str(userA) + " and user:" + str(userB) + " (using Pearson similarity) is: " + str(similarity) + " normal" + str(similarity_w) + " weighted")
        print("________________________________________________________________________________________________\n")
        
    
    def example_user_based_collaborative_filtering(self):
        # Shows for a random user, the 10 most similar users
        randomId = random.randint(1, self.ds_manager.get_users_count())
        self.user_based_collaborative_filtering.show_top_x_similar_users(randomId, 10)

        # Shows the top 10 predictions for the same user
        # Neighbourhood size = 50
        start_time = time.time()
        self.user_based_collaborative_filtering.show_top_x_recommendations(randomId, 10, 50)
        end_time = time.time()
        print("Time taken to calculate predictions:", end_time - start_time, "seconds")
        
        

if __name__ == "__main__":
    RecommendationAssignment()
