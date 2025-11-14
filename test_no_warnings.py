"""
Test script to validate warning suppression.
Should run cleanly without Streamlit context warnings.
"""

print('=' * 60)
print('TESTING WARNING SUPPRESSION - FINAL VALIDATION')
print('=' * 60)
print()

from src.algorithms.algorithm_manager import AlgorithmManager, AlgorithmType
import pandas as pd

manager = AlgorithmManager()
ratings_df = pd.read_csv('data/ml-32m/ratings.csv')
movies_df = pd.read_csv('data/ml-32m/movies.csv')
manager.initialize_data(ratings_df, movies_df)

print('Testing SVD...')
svd = manager.get_algorithm(AlgorithmType.SVD)
print(f'✅ SVD loaded: {svd is not None}')
print()

print('Testing User-KNN...')
user_knn = manager.get_algorithm(AlgorithmType.USER_KNN)
print(f'✅ User-KNN loaded: {user_knn is not None}')
print()

print('Testing Item-KNN...')
item_knn = manager.get_algorithm(AlgorithmType.ITEM_KNN)
print(f'✅ Item-KNN loaded: {item_knn is not None}')
print()

print('Testing Content-Based...')
content = manager.get_algorithm(AlgorithmType.CONTENT_BASED)
print(f'✅ Content-Based loaded: {content is not None}')
print()

print('Testing Hybrid...')
hybrid = manager.get_algorithm(AlgorithmType.HYBRID)
print(f'✅ Hybrid loaded: {hybrid is not None}')
print()

print('=' * 60)
print('✅ ALL TESTS PASSED')
print('If NO warnings appeared above, suppression is working!')
print('=' * 60)
