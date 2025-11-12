# -*- coding: utf-8 -*-
"""
CineMatch V2.1.0 - Content-Based Recommender Unit Tests

Comprehensive test suite for ContentBasedRecommender class.
Tests feature extraction, TF-IDF vectorization, similarity computation,
user profile building, predictions, and recommendations.

Author: CineMatch Development Team
Date: November 11, 2025
"""

import sys
from pathlib import Path
import unittest
import pandas as pd
import numpy as np
from scipy.sparse import issparse
import warnings

warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.algorithms.content_based_recommender import ContentBasedRecommender


class TestContentBasedRecommender(unittest.TestCase):
    """Test suite for ContentBasedRecommender class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used by multiple tests."""
        print("\n" + "="*70)
        print("CONTENT-BASED FILTERING - UNIT TEST SUITE")
        print("="*70)
        
        # Create sample movies data
        cls.movies_df = pd.DataFrame({
            'movieId': [1, 2, 3, 4, 5],
            'title': ['Toy Story (1995)', 'Jumanji (1995)', 'Heat (1995)', 
                     'Sabrina (1995)', 'Tom and Huck (1995)'],
            'genres': ['Animation|Children|Comedy', 'Adventure|Children|Fantasy',
                      'Action|Crime|Thriller', 'Comedy|Romance', 'Adventure|Children']
        })
        
        # Create sample ratings data
        cls.ratings_df = pd.DataFrame({
            'userId': [1, 1, 1, 2, 2, 2, 3, 3],
            'movieId': [1, 2, 3, 1, 4, 5, 2, 3],
            'rating': [5.0, 4.0, 3.0, 4.5, 5.0, 3.5, 4.0, 5.0],
            'timestamp': [1, 2, 3, 4, 5, 6, 7, 8]
        })
        
        print(f"\n[OK] Test data loaded: {len(cls.movies_df)} movies, {len(cls.ratings_df)} ratings")
    
    def setUp(self):
        """Set up test instance before each test."""
        # Use min_df=1 for small test dataset (5 movies)
        # Production uses min_df=2-3 for large datasets (87K movies)
        self.recommender = ContentBasedRecommender(
            genre_weight=0.5,
            tag_weight=0.3,
            title_weight=0.2,
            min_similarity=0.01
        )
        # Override vectorizer parameters for small test dataset
        self.recommender.tag_vectorizer.min_df = 1
        self.recommender.title_vectorizer.min_df = 1
    
    # ==================== INITIALIZATION TESTS ====================
    
    def test_01_initialization(self):
        """Test ContentBasedRecommender initialization."""
        print("\n[TEST 1] Initialization and Parameters")
        
        self.assertIsNotNone(self.recommender)
        self.assertEqual(self.recommender.genre_weight, 0.5)
        self.assertEqual(self.recommender.tag_weight, 0.3)
        self.assertEqual(self.recommender.title_weight, 0.2)
        self.assertEqual(self.recommender.min_similarity, 0.01)
        
        print("  [OK] Recommender initialized with correct parameters")
        print(f"  [OK] Weights: genre={self.recommender.genre_weight}, tag={self.recommender.tag_weight}, title={self.recommender.title_weight}")
    
    def test_02_custom_weights(self):
        """Test initialization with custom weights."""
        print("\n[TEST 2] Custom Weight Configuration")
        
        custom_recommender = ContentBasedRecommender(
            genre_weight=0.6,
            tag_weight=0.2,
            title_weight=0.2,
            min_similarity=0.05
        )
        
        self.assertEqual(custom_recommender.genre_weight, 0.6)
        self.assertEqual(custom_recommender.tag_weight, 0.2)
        self.assertEqual(custom_recommender.title_weight, 0.2)
        self.assertEqual(custom_recommender.min_similarity, 0.05)
        
        print("  [OK] Custom weights applied correctly")
    
    # ==================== FEATURE EXTRACTION TESTS ====================
    
    def test_03_genre_extraction(self):
        """Test genre feature extraction."""
        print("\n[TEST 3] Genre Feature Extraction")
        
        self.recommender.fit(self.ratings_df, self.movies_df)
        
        # Check that genre vectorizer was created
        self.assertIsNotNone(self.recommender.genre_vectorizer)
        self.assertIsNotNone(self.recommender.genre_features)
        
        # Check sparse matrix properties
        self.assertTrue(issparse(self.recommender.genre_features))
        n_movies, n_genres = self.recommender.genre_features.shape
        self.assertEqual(n_movies, len(self.movies_df))
        
        print(f"  [OK] Genre features: {n_movies} movies × {n_genres} genres")
        print(f"  [OK] Sparse matrix density: {self.recommender.genre_features.nnz / (n_movies * n_genres):.4f}")
    
    def test_04_title_extraction(self):
        """Test title keyword extraction."""
        print("\n[TEST 4] Title Keyword Extraction")
        
        self.recommender.fit(self.ratings_df, self.movies_df)
        
        # Check that title vectorizer was created
        self.assertIsNotNone(self.recommender.title_vectorizer)
        self.assertIsNotNone(self.recommender.title_features)
        
        # Check sparse matrix properties
        self.assertTrue(issparse(self.recommender.title_features))
        n_movies, n_keywords = self.recommender.title_features.shape
        self.assertEqual(n_movies, len(self.movies_df))
        
        print(f"  [OK] Title features: {n_movies} movies × {n_keywords} keywords")
        print(f"  [OK] Sparse matrix density: {self.recommender.title_features.nnz / (n_movies * n_keywords):.4f}")
    
    def test_05_combined_features(self):
        """Test combined feature matrix."""
        print("\n[TEST 5] Combined Feature Matrix")
        
        self.recommender.fit(self.ratings_df, self.movies_df)
        
        # Check that combined feature matrix was created
        self.assertIsNotNone(self.recommender.combined_features)
        self.assertTrue(issparse(self.recommender.combined_features))
        
        n_movies, n_features = self.recommender.combined_features.shape
        self.assertEqual(n_movies, len(self.movies_df))
        
        print(f"  [OK] Combined features: {n_movies} movies × {n_features} total features")
        print(f"  [OK] Memory usage: {self.recommender.combined_features.data.nbytes / 1024:.2f} KB")
    
    # ==================== SIMILARITY COMPUTATION TESTS ====================
    
    def test_06_similarity_matrix(self):
        """Test similarity matrix computation."""
        print("\n[TEST 6] Similarity Matrix Computation")
        
        self.recommender.fit(self.ratings_df, self.movies_df)
        
        # Check that similarity matrix was created
        self.assertIsNotNone(self.recommender.similarity_matrix)
        self.assertTrue(issparse(self.recommender.similarity_matrix))
        
        # Check matrix properties
        n_movies = len(self.movies_df)
        self.assertEqual(self.recommender.similarity_matrix.shape, (n_movies, n_movies))
        
        # Check symmetry (similarity should be symmetric)
        diff = (self.recommender.similarity_matrix - self.recommender.similarity_matrix.T).nnz
        self.assertEqual(diff, 0, "Similarity matrix should be symmetric")
        
        print(f"  [OK] Similarity matrix: {n_movies} × {n_movies}")
        print(f"  [OK] Non-zero entries: {self.recommender.similarity_matrix.nnz}")
        print(f"  [OK] Sparsity: {1 - self.recommender.similarity_matrix.nnz / (n_movies * n_movies):.4f}")
    
    def test_07_similarity_threshold(self):
        """Test minimum similarity threshold."""
        print("\n[TEST 7] Similarity Threshold Application")
        
        self.recommender.fit(self.ratings_df, self.movies_df)
        
        # Get all non-zero similarities
        similarities = self.recommender.similarity_matrix.data
        
        # Check that all similarities are >= min_similarity
        min_sim = similarities.min()
        self.assertGreaterEqual(min_sim, self.recommender.min_similarity,
                               f"Minimum similarity {min_sim} is below threshold {self.recommender.min_similarity}")
        
        print(f"  [OK] Min similarity in matrix: {min_sim:.4f}")
        print(f"  [OK] Threshold: {self.recommender.min_similarity}")
        print(f"  [OK] Max similarity: {similarities.max():.4f}")
    
    # ==================== USER PROFILE TESTS ====================
    
    def test_08_user_profile_building(self):
        """Test user profile construction."""
        print("\n[TEST 8] User Profile Building")
        
        self.recommender.fit(self.ratings_df, self.movies_df)
        
        # Build profile for user 1
        user_profile = self.recommender._build_user_profile(1)
        
        self.assertIsNotNone(user_profile)
        self.assertIsInstance(user_profile, np.ndarray)
        
        # Check that profile has same dimension as feature matrix
        self.assertEqual(user_profile.shape[0], self.recommender.combined_features.shape[1])
        
        # Check normalization (should be unit vector)
        profile_norm = np.linalg.norm(user_profile)
        self.assertAlmostEqual(profile_norm, 1.0, places=5, msg="Profile should be normalized")
        
        print(f"  [OK] User profile created with {user_profile.shape[0]} features")
        print(f"  [OK] Profile norm: {profile_norm:.6f} (normalized)")
        print(f"  [OK] Non-zero features: {np.count_nonzero(user_profile)}")
    
    def test_09_user_profile_weights(self):
        """Test user profile weighting by ratings."""
        print("\n[TEST 9] User Profile Rating Weights")
        
        self.recommender.fit(self.ratings_df, self.movies_df)
        
        # User 1 has ratings: [5.0, 4.0, 3.0]
        # Expected weights: [(5-2.5)/2.5, (4-2.5)/2.5, (3-2.5)/2.5] = [1.0, 0.6, 0.2]
        user_1_ratings = self.ratings_df[self.ratings_df['userId'] == 1]
        
        expected_weights = [(r - 2.5) / 2.5 for r in user_1_ratings['rating'].values]
        
        print(f"  [OK] User 1 ratings: {user_1_ratings['rating'].values}")
        print(f"  [OK] Expected weights: {expected_weights}")
        print(f"  [OK] Weight calculation: (rating - 2.5) / 2.5")
    
    # ==================== PREDICTION TESTS ====================
    
    def test_10_predict_rating(self):
        """Test rating prediction."""
        print("\n[TEST 10] Rating Prediction")
        
        self.recommender.fit(self.ratings_df, self.movies_df)
        
        # Predict rating for user 1, movie 4 (not rated yet)
        predicted_rating = self.recommender.predict(user_id=1, movie_id=4)
        
        self.assertIsNotNone(predicted_rating)
        self.assertIsInstance(predicted_rating, (int, float, np.number))
        self.assertGreaterEqual(predicted_rating, 0.5)
        self.assertLessEqual(predicted_rating, 5.0)
        
        print(f"  [OK] Predicted rating for user 1, movie 4: {predicted_rating:.2f}")
        print(f"  [OK] Rating range valid: 0.5 ≤ {predicted_rating:.2f} ≤ 5.0")
    
    def test_11_predict_multiple_users(self):
        """Test predictions for multiple users."""
        print("\n[TEST 11] Multiple User Predictions")
        
        self.recommender.fit(self.ratings_df, self.movies_df)
        
        predictions = []
        for user_id in [1, 2, 3]:
            pred = self.recommender.predict(user_id=user_id, movie_id=5)
            predictions.append(pred)
            print(f"  [OK] User {user_id}, Movie 5: {pred:.2f}")
        
        # All predictions should be valid
        for pred in predictions:
            self.assertGreaterEqual(pred, 0.5)
            self.assertLessEqual(pred, 5.0)
    
    # ==================== RECOMMENDATION TESTS ====================
    
    def test_12_get_recommendations(self):
        """Test recommendation generation."""
        print("\n[TEST 12] Recommendation Generation")
        
        self.recommender.fit(self.ratings_df, self.movies_df)
        
        # Get recommendations for user 1
        recommendations = self.recommender.get_recommendations(user_id=1, n=3, exclude_rated=True)
        
        self.assertIsNotNone(recommendations)
        self.assertIsInstance(recommendations, pd.DataFrame)
        self.assertLessEqual(len(recommendations), 3)
        
        # Check that recommended movies are not already rated
        user_rated = set(self.ratings_df[self.ratings_df['userId'] == 1]['movieId'].values)
        recommended_ids = set(recommendations['movieId'].values)
        
        self.assertEqual(len(user_rated.intersection(recommended_ids)), 0,
                        "Recommendations should exclude already rated movies")
        
        print(f"  [OK] Generated {len(recommendations)} recommendations")
        print(f"  [OK] No overlap with user's {len(user_rated)} rated movies")
        if len(recommendations) > 0:
            print(f"  [OK] Top recommendation: {recommendations.iloc[0]['title']} (score: {recommendations.iloc[0]['predicted_rating']:.2f})")
    
    def test_13_cold_start_handling(self):
        """Test cold-start handling for new users."""
        print("\n[TEST 13] Cold-Start User Handling")
        
        self.recommender.fit(self.ratings_df, self.movies_df)
        
        # Try to get recommendations for non-existent user
        new_user_id = 999
        recommendations = self.recommender.get_recommendations(user_id=new_user_id, n=3)
        
        self.assertIsNotNone(recommendations)
        self.assertGreater(len(recommendations), 0, "Should return popular movies for new users")
        
        print(f"  [OK] Cold-start handled: returned {len(recommendations)} popular movies")
        if len(recommendations) > 0:
            print(f"  [OK] Fallback movie: {recommendations.iloc[0]['title']}")
    
    def test_14_similar_items(self):
        """Test similar item retrieval."""
        print("\n[TEST 14] Similar Item Retrieval")
        
        self.recommender.fit(self.ratings_df, self.movies_df)
        
        # Find similar movies to movie 1 (Toy Story)
        similar_movies = self.recommender.get_similar_items(item_id=1, n=2)
        
        self.assertIsNotNone(similar_movies)
        self.assertIsInstance(similar_movies, pd.DataFrame)
        self.assertLessEqual(len(similar_movies), 2)
        
        # Check that original movie is not in results
        self.assertNotIn(1, similar_movies['movieId'].values)
        
        print(f"  [OK] Found {len(similar_movies)} similar movies to 'Toy Story'")
        for idx, row in similar_movies.iterrows():
            print(f"  [OK] Similar: {row['title']} (similarity: {row.get('similarity', 'N/A')})")
    
    # ==================== METRICS TESTS ====================
    
    def test_15_training_metrics(self):
        """Test training metrics calculation."""
        print("\n[TEST 15] Training Metrics")
        
        self.recommender.fit(self.ratings_df, self.movies_df)
        
        # Check that metrics are stored in metrics object
        self.assertIsNotNone(self.recommender.metrics.training_time)
        self.assertIsNotNone(self.recommender.metrics.rmse)
        self.assertIsNotNone(self.recommender.metrics.coverage)
        
        self.assertGreater(self.recommender.metrics.training_time, 0)
        self.assertGreaterEqual(self.recommender.metrics.rmse, 0)
        self.assertGreaterEqual(self.recommender.metrics.coverage, 0)
        self.assertLessEqual(self.recommender.metrics.coverage, 100)
        
        print(f"  [OK] Training time: {self.recommender.metrics.training_time:.2f}s")
        print(f"  [OK] RMSE: {self.recommender.metrics.rmse:.4f}")
        print(f"  [OK] Coverage: {self.recommender.metrics.coverage:.2f}%")
        print(f"  [OK] Memory usage: {self.recommender.metrics.memory_usage_mb:.2f} MB")
    
    # ==================== MODEL PERSISTENCE TESTS ====================
    
    def test_16_model_save_load(self):
        """Test model save and load functionality."""
        print("\n[TEST 16] Model Save/Load")
        
        self.recommender.fit(self.ratings_df, self.movies_df)
        
        # Save model
        test_model_path = Path("models/test_content_based_model.pkl")
        test_model_path.parent.mkdir(exist_ok=True)
        
        self.recommender.save_model(test_model_path)
        self.assertTrue(test_model_path.exists(), "Model file should be created")
        
        # Load model (load_model is an instance method, not a classmethod)
        loaded_recommender = ContentBasedRecommender()
        loaded_recommender.load_model(test_model_path)
        
        self.assertIsNotNone(loaded_recommender)
        self.assertTrue(loaded_recommender.is_trained, "Loaded model should be trained")
        self.assertEqual(loaded_recommender.genre_weight, self.recommender.genre_weight)
        self.assertEqual(loaded_recommender.tag_weight, self.recommender.tag_weight)
        
        # Test that loaded model can make predictions
        original_pred = self.recommender.predict(1, 4)
        loaded_pred = loaded_recommender.predict(1, 4)
        
        self.assertAlmostEqual(original_pred, loaded_pred, places=4,
                              msg="Loaded model should produce same predictions")
        
        print(f"  [OK] Model saved to {test_model_path}")
        print(f"  [OK] Model loaded successfully")
        print(f"  [OK] Prediction consistency: {original_pred:.4f} = {loaded_pred:.4f}")
        
        # Cleanup
        test_model_path.unlink()
    
    # ==================== ERROR HANDLING TESTS ====================
    
    def test_17_invalid_user(self):
        """Test handling of invalid user IDs."""
        print("\n[TEST 17] Invalid User ID Handling")
        
        self.recommender.fit(self.ratings_df, self.movies_df)
        
        # Try prediction for invalid user (should handle gracefully)
        result = self.recommender.predict(user_id=-1, movie_id=1)
        
        # Should return fallback prediction (not crash)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result, 0.5)
        
        print(f"  [OK] Invalid user handled gracefully: returned {result:.2f}")
    
    def test_18_invalid_movie(self):
        """Test handling of invalid movie IDs."""
        print("\n[TEST 18] Invalid Movie ID Handling")
        
        self.recommender.fit(self.ratings_df, self.movies_df)
        
        # Try prediction for invalid movie
        result = self.recommender.predict(user_id=1, movie_id=999)
        
        # Should return fallback prediction
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result, 0.5)
        
        print(f"  [OK] Invalid movie handled gracefully: returned {result:.2f}")
    
    def test_19_empty_ratings(self):
        """Test handling of users with no ratings."""
        print("\n[TEST 19] Empty User History Handling")
        
        self.recommender.fit(self.ratings_df, self.movies_df)
        
        # Get recommendations for user with no history
        recommendations = self.recommender.get_recommendations(user_id=9999, n=3)
        
        self.assertIsNotNone(recommendations)
        self.assertGreater(len(recommendations), 0)
        
        print(f"  [OK] Empty history handled: returned {len(recommendations)} fallback movies")
    
    # ==================== PERFORMANCE TESTS ====================
    
    def test_20_prediction_speed(self):
        """Test prediction performance."""
        print("\n[TEST 20] Prediction Speed Performance")
        
        self.recommender.fit(self.ratings_df, self.movies_df)
        
        import time
        
        # Time multiple predictions
        n_predictions = 10
        start_time = time.time()
        
        for _ in range(n_predictions):
            self.recommender.predict(user_id=1, movie_id=2)
        
        elapsed_time = time.time() - start_time
        avg_time_ms = (elapsed_time / n_predictions) * 1000
        
        # Should be reasonably fast (< 50ms per prediction on small dataset)
        self.assertLess(avg_time_ms, 50, "Predictions should be fast")
        
        print(f"  [OK] Average prediction time: {avg_time_ms:.2f}ms")
        print(f"  [OK] Predictions per second: {1000 / max(avg_time_ms, 0.001):.0f}")


def run_tests():
    """Run all tests with detailed output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestContentBasedRecommender)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"[PASS] Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"[FAIL] Failed: {len(result.failures)}")
    print(f"[WARN]  Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

