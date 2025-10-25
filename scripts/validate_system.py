"""
CineMatch V1.0.0 - Pre-Launch Validation Script

Validates entire system before thesis defense.
Checks files, dependencies, model, data, and features.

Author: CineMatch Team
Date: October 24, 2025
"""

import sys
from pathlib import Path
import importlib.util

def check_mark(status: bool) -> str:
    """Return check mark or X based on status"""
    return "✅" if status else "❌"

def print_section(title: str):
    """Print formatted section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

class SystemValidator:
    """Validates entire CineMatch system"""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        
    def log_issue(self, message: str):
        """Log a critical issue"""
        self.issues.append(message)
        print(f"❌ ISSUE: {message}")
        
    def log_warning(self, message: str):
        """Log a warning"""
        self.warnings.append(message)
        print(f"⚠️  WARNING: {message}")
        
    def log_success(self, message: str):
        """Log a success"""
        print(f"✅ {message}")
        
    def validate_project_structure(self):
        """Check if all required directories exist"""
        print_section("1. Project Structure Validation")
        
        required_dirs = [
            "data/ml-32m",
            "models",
            "src",
            "app",
            "app/pages",
            "scripts"
        ]
        
        all_good = True
        for dir_path in required_dirs:
            path = Path(dir_path)
            if path.exists():
                self.log_success(f"Directory exists: {dir_path}")
            else:
                self.log_issue(f"Missing directory: {dir_path}")
                all_good = False
        
        return all_good
    
    def validate_dataset(self):
        """Check if dataset files exist"""
        print_section("2. Dataset Validation")
        
        required_files = [
            "data/ml-32m/ratings.csv",
            "data/ml-32m/movies.csv",
            "data/ml-32m/links.csv",
            "data/ml-32m/tags.csv"
        ]
        
        all_good = True
        for file_path in required_files:
            path = Path(file_path)
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                self.log_success(f"Dataset file: {file_path} ({size_mb:.1f} MB)")
            else:
                self.log_issue(f"Missing dataset file: {file_path}")
                all_good = False
        
        return all_good
    
    def validate_dependencies(self):
        """Check if all Python dependencies are installed"""
        print_section("3. Python Dependencies Validation")
        
        required_packages = [
            "pandas",
            "numpy",
            "streamlit",
            "plotly",
            "joblib",
            "scipy",
            "sklearn"  # scikit-learn
        ]
        
        all_good = True
        for package in required_packages:
            spec = importlib.util.find_spec(package)
            if spec is not None:
                self.log_success(f"Package installed: {package}")
            else:
                self.log_issue(f"Missing package: {package}")
                all_good = False
        
        # Check for optional surprise package
        spec = importlib.util.find_spec("surprise")
        if spec is None:
            self.log_warning("scikit-surprise not installed (using sklearn alternative)")
        else:
            self.log_success("Package installed: scikit-surprise")
        
        return all_good
    
    def validate_source_code(self):
        """Check if all source code files exist"""
        print_section("4. Source Code Validation")
        
        required_files = [
            "src/data_processing.py",
            "src/model_training.py",
            "src/recommendation_engine.py",
            "src/utils.py",
            "app/main.py",
            "app/pages/1_🏠_Home.py",
            "app/pages/2_🎬_Recommend.py",
            "app/pages/3_📊_Analytics.py"
        ]
        
        # Optional sklearn files
        optional_files = [
            "src/model_training_sklearn.py",
            "src/recommendation_engine_sklearn.py"
        ]
        
        all_good = True
        for file_path in required_files:
            path = Path(file_path)
            if path.exists():
                self.log_success(f"Source file: {file_path}")
            else:
                self.log_issue(f"Missing source file: {file_path}")
                all_good = False
        
        for file_path in optional_files:
            path = Path(file_path)
            if path.exists():
                self.log_success(f"Optional file: {file_path}")
            else:
                self.log_warning(f"Optional file not found: {file_path}")
        
        return all_good
    
    def validate_model(self):
        """Check if trained model exists"""
        print_section("5. Model Validation")
        
        sklearn_model = Path("models/svd_model_sklearn.pkl")
        surprise_model = Path("models/svd_model.pkl")
        
        if sklearn_model.exists():
            size_mb = sklearn_model.stat().st_size / (1024 * 1024)
            self.log_success(f"Trained model found: svd_model_sklearn.pkl ({size_mb:.1f} MB)")
            return True
        elif surprise_model.exists():
            size_mb = surprise_model.stat().st_size / (1024 * 1024)
            self.log_success(f"Trained model found: svd_model.pkl ({size_mb:.1f} MB)")
            return True
        else:
            self.log_issue("No trained model found! Run: python src/model_training_sklearn.py")
            return False
    
    def validate_documentation(self):
        """Check if documentation files exist"""
        print_section("6. Documentation Validation")
        
        required_docs = [
            "README.md",
            "QUICKSTART.md",
            "ARCHITECTURE.md",
            "PROJECT_SUMMARY.md",
            "requirements.txt"
        ]
        
        all_good = True
        for doc in required_docs:
            path = Path(doc)
            if path.exists():
                self.log_success(f"Documentation: {doc}")
            else:
                self.log_warning(f"Missing documentation: {doc}")
                all_good = False
        
        return all_good
    
    def validate_docker(self):
        """Check if Docker configuration exists"""
        print_section("7. Docker Configuration Validation")
        
        dockerfile = Path("Dockerfile")
        compose = Path("docker-compose.yml")
        
        all_good = True
        if dockerfile.exists():
            self.log_success("Dockerfile found")
        else:
            self.log_warning("Dockerfile not found (optional for local dev)")
            all_good = False
        
        if compose.exists():
            self.log_success("docker-compose.yml found")
        else:
            self.log_warning("docker-compose.yml not found (optional for local dev)")
            all_good = False
        
        return all_good
    
    def validate_scripts(self):
        """Check if automation scripts exist"""
        print_section("8. Automation Scripts Validation")
        
        scripts = [
            "start.bat",
            "scripts/test_features.py",
            "scripts/demo_script.md"
        ]
        
        for script in scripts:
            path = Path(script)
            if path.exists():
                self.log_success(f"Script found: {script}")
            else:
                self.log_warning(f"Script not found: {script}")
        
        return True  # Scripts are optional
    
    def test_import_engine(self):
        """Try to import recommendation engine"""
        print_section("9. Engine Import Test")
        
        try:
            sys.path.append(str(Path.cwd()))
            
            # Try sklearn version first
            try:
                from src.recommendation_engine_sklearn import RecommendationEngine
                self.log_success("Successfully imported sklearn-based recommendation engine")
                return True
            except ImportError:
                # Try original version
                from src.recommendation_engine import RecommendationEngine
                self.log_success("Successfully imported original recommendation engine")
                return True
        except Exception as e:
            self.log_issue(f"Failed to import recommendation engine: {e}")
            return False
    
    def test_data_loading(self):
        """Try to load data"""
        print_section("10. Data Loading Test")
        
        try:
            sys.path.append(str(Path.cwd()))
            from src.data_processing import load_ratings, load_movies, check_data_integrity
            
            # Test integrity check
            try:
                integrity_result = check_data_integrity()
                if isinstance(integrity_result, tuple):
                    integrity_ok, message = integrity_result
                else:
                    integrity_ok = integrity_result
                    message = "Data integrity check completed"
                
                if integrity_ok:
                    self.log_success(f"Data integrity check passed: {message}")
                else:
                    self.log_warning(f"Data integrity check: {message}")
            except Exception as e:
                self.log_warning(f"Data integrity check: {e}")
            
            # Try loading small sample
            ratings = load_ratings(sample_size=1000)
            movies = load_movies()
            
            self.log_success(f"Loaded {len(ratings)} ratings (sample)")
            self.log_success(f"Loaded {len(movies)} movies")
            
            return True
        except Exception as e:
            self.log_issue(f"Failed to load data: {e}")
            return False
    
    def run_all_validations(self):
        """Run all validation checks"""
        print("\n" + "="*70)
        print("  CineMatch V1.0.0 - System Validation")
        print("="*70)
        
        results = {
            "Project Structure": self.validate_project_structure(),
            "Dataset": self.validate_dataset(),
            "Dependencies": self.validate_dependencies(),
            "Source Code": self.validate_source_code(),
            "Model": self.validate_model(),
            "Documentation": self.validate_documentation(),
            "Docker": self.validate_docker(),
            "Scripts": self.validate_scripts(),
            "Engine Import": self.test_import_engine(),
            "Data Loading": self.test_data_loading()
        }
        
        # Summary
        print_section("VALIDATION SUMMARY")
        
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        for category, status in results.items():
            print(f"{check_mark(status)} {category}")
        
        print(f"\n{'='*70}")
        print(f"  Score: {passed}/{total} checks passed ({passed/total*100:.1f}%)")
        print(f"{'='*70}\n")
        
        if self.issues:
            print("🔴 CRITICAL ISSUES:")
            for issue in self.issues:
                print(f"  • {issue}")
            print()
        
        if self.warnings:
            print("🟡 WARNINGS:")
            for warning in self.warnings:
                print(f"  • {warning}")
            print()
        
        if passed == total and not self.issues:
            print("🎉 ALL VALIDATIONS PASSED!")
            print("✅ System is ready for thesis defense!")
            print("\nNext steps:")
            print("  1. Run: streamlit run app/main.py")
            print("  2. Test all features")
            print("  3. Review demo_script.md")
            return True
        else:
            print("⚠️  VALIDATION INCOMPLETE")
            print(f"   {len(self.issues)} critical issue(s)")
            print(f"   {len(self.warnings)} warning(s)")
            print("\nPlease resolve issues before proceeding.")
            return False


if __name__ == "__main__":
    validator = SystemValidator()
    success = validator.run_all_validations()
    sys.exit(0 if success else 1)
