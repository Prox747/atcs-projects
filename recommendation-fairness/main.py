import pandas as pd
import os

# Ottieni il percorso della cartella del progetto
project_folder = os.path.dirname(__file__)


# Costruisci i percorsi completi dei file CSV
links_file_path = os.path.join(project_folder, 'ml-latest-small', 'links.csv')
movies_file_path = os.path.join(project_folder, 'ml-latest-small', 'movies.csv')
ratings_file_path = os.path.join(project_folder, 'ml-latest-small', 'ratings.csv')
tags_file_path = os.path.join(project_folder, 'ml-latest-small', 'tags.csv')


# Leggi i file CSV
links_df = pd.read_csv(links_file_path)
movies_df = pd.read_csv(movies_file_path)
ratings_df = pd.read_csv(ratings_file_path)
tags_df = pd.read_csv(tags_file_path)

 
# Visualizza le prime righe di ciascun DataFrame
# e mostra il numero delle righe per ciascun DataFrame
print("\n-------- LINKS.CSV --------")
print("Numero di elementi in links.csv:", len(links_df))
print("Primi elementi di links.csv:")
print(links_df.head())


print("\n-------- MOVIES.CSV --------")
print("Numero di elementi in movies.csv:", len(movies_df))
print("Primi elementi di movies.csv:")
print(movies_df.head())

print("\n-------- RATINGS.CSV --------")
print("Numero di elementi in ratings.csv:", len(ratings_df))
print("Primi elementi di ratings.csv:")
print(ratings_df.head())

print("\n-------- TAGS.CSV --------")
print("Numero di elementi in tags.csv:", len(tags_df))
print("Primi elementi di tags.csv:")
print(tags_df.head())
