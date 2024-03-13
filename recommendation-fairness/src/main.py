import pandas as pd
import random as rand
import os
from df_manager import DataFrameManager
from recomm_techniques.similarity_calc import SimCalc



class RecommendationAssignment:
    def __init__(self):
        self.df_manager = DataFrameManager()
        self.sim_calc = SimCalc(self.df_manager)
        
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
        
        print("\n#####################################################")

    
    def example_similarity(self, userA: int, userB: int):
        similarity = self.sim_calc.pearson_correlation(userA, userB)
        print("\n\n_______________________EXAMPLE PEARSON SIMILARITY BETWEEN TWO USERS_____________________________")
        print("\nThe similarity between user:" + str(userA) + " and user:" + str(userB) + " (using Pearson similarity) is: " + str(similarity))
        print("________________________________________________________________________________________________\n")


if __name__ == "__main__":
    RecommendationAssignment()
