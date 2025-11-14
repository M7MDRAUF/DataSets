"""Quick test of SVD only - validate no warnings"""
from src.algorithms.algorithm_manager import AlgorithmManager, AlgorithmType
import pandas as pd

print('Testing SVD warning suppression...\n')

manager = AlgorithmManager()
ratings_df = pd.read_csv('data/ml-32m/ratings.csv')
movies_df = pd.read_csv('data/ml-32m/movies.csv')
manager.initialize_data(ratings_df, movies_df)

svd = manager.get_algorithm(AlgorithmType.SVD)
print(f'\n✅ SVD loaded successfully')
print('If NO ScriptRunContext warning above, suppression is WORKING!')
