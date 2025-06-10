import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
import argparse
import sys
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class KNNTrainer:
    """Main class for training KNN models with automatic preprocessing"""
    
    def __init__(self, input_file, target_column=None, n_neighbors=5, 
                 weights='distance', test_size=0.2, random_state=42):
        self.input_file = input_file
        self.target_column = target_column
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.preprocessor = None
        self.metadata = {}
        
    def load_data(self):
        """Load and validate CSV data"""
        try:
            # Load data
            self.df = pd.read_csv(self.input_file)
            print(f"\nSuccessfully loaded data from {self.input_file}")
            print(f"  Shape: {self.df.shape}")
            print(f"  Memory usage: {self.df.memory_usage().sum() / 1024**2:.2f} MB")
            
            # Check for empty dataset
            if self.df.empty:
                raise ValueError("The dataset is empty")
            # Display basic statistics
            print(f"\nDataset Overview:")
            print(f"  - Total samples: {len(self.df)}")
            print(f"  - Total features: {len(self.df.columns)}")
            print(f"  - Missing values: {self.df.isnull().sum().sum()}")
            
            return True
            
        except FileNotFoundError:
            print(f"Error: File '{self.input_file}' not found")
            return False
        except pd.errors.EmptyDataError:
            print(f"Error: File '{self.input_file}' is empty")
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def select_target_column(self):
        """Interactive target column selection"""
        if self.target_column:
            if self.target_column not in self.df.columns:
                print(f"Error: Target column '{self.target_column}' not found")
                self.target_column = None
            else:
                return True
                
        # Display columns for selection
        print("\nAvailable columns:")
        print("-" * 50)
        for i, col in enumerate(self.df.columns):
            dtype = self.df[col].dtype
            unique_count = self.df[col].nunique()
            null_count = self.df[col].isnull().sum()
            
            print(f"{i:2d}: {col:<20} | Type: {str(dtype):<10} | "
                  f"Unique: {unique_count:<5} | Missing: {null_count}")
        
        # Get user selection
        while True:
            try:
                choice = input("\nEnter the column number for the target variable: ")
                col_idx = int(choice)
                if 0 <= col_idx < len(self.df.columns):
                    self.target_column = self.df.columns[col_idx]
                    print(f"\nSelected '{self.target_column}' as target column")
                    
                    # Display target distribution
                    print(f"\nTarget Distribution:")
                    print(self.df[self.target_column].value_counts())
                    
                    return True
                else:
                    print("Invalid column number. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
            except KeyboardInterrupt:
                print("\n\nOperation cancelled by user")
                return False
    
    def preprocess_data(self):
        """Preprocess data with automatic handling of different data types"""
        print("\nPreprocessing data...")
        
        # Separate features and target
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        
        # Handle missing values in target
        if y.isnull().any():
            print(f"  Removing {y.isnull().sum()} samples with missing target values")
            mask = ~y.isnull()
            X = X[mask]
            y = y[mask]
        
        # Identify column types
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"  Numeric features ({len(numeric_features)}): {numeric_features[:5]}{'...' if len(numeric_features) > 5 else ''}")
        print(f"  Categorical features ({len(categorical_features)}): {categorical_features[:5]}{'...' if len(categorical_features) > 5 else ''}")
        
        # Create preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Store metadata
        self.metadata['numeric_features'] = numeric_features
        self.metadata['categorical_features'] = categorical_features
        self.metadata['n_features_original'] = len(X.columns)
        self.metadata['target_classes'] = y.unique().tolist()
        
        return X, y
    
    def split_and_train(self, X, y):
        """Split data and train the model"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"\nData Split:")
        print(f"  Training set: {len(X_train)} samples")
        print(f"  Test set: {len(X_test)} samples")
        
        # Create and train pipeline
        print(f"\nTraining KNN model...")
        print(f"  Parameters: k={self.n_neighbors}, weights={self.weights}")
        
        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', KNeighborsClassifier(
                n_neighbors=self.n_neighbors,
                weights=self.weights,
                n_jobs=-1
            ))
        ])
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Get transformed feature count
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        self.metadata['n_features_transformed'] = X_train_transformed.shape[1]
        
        print(f"  Model trained successfully")
        print(f"  Features after preprocessing: {self.metadata['n_features_transformed']}")
        
        return X_test, y_test
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model"""
        print("\nModel Evaluation:")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report
        print("\n  Detailed Classification Report:")
        print("  " + "-" * 60)
        report = classification_report(y_test, y_pred)
        for line in report.split('\n'):
            print(f"  {line}")
        
        # Confusion matrix
        print("\n  Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        classes = np.unique(y_test)
        
        # Print header
        print(f"  {'Actual\\Predicted':<20}", end='')
        for cls in classes:
            print(f"{str(cls):<15}", end='')
        print()
        
        # Print matrix
        for i, cls in enumerate(classes):
            print(f"  {str(cls):<20}", end='')
            for j in range(len(classes)):
                print(f"{cm[i, j]:<15}", end='')
            print()
        
        # Store evaluation metrics
        self.metadata['accuracy'] = float(accuracy)
        self.metadata['test_size'] = len(y_test)
        
        return accuracy
    
    def save_model(self, output_path):
        """Save model and metadata"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Add timestamp and training info to metadata
        self.metadata['training_date'] = datetime.now().isoformat()
        self.metadata['input_file'] = self.input_file
        self.metadata['target_column'] = self.target_column
        self.metadata['model_params'] = {
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'test_size': self.test_size,
            'random_state': self.random_state
        }
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'metadata': self.metadata
        }
        
        joblib.dump(model_data, output_path)
        print(f"\nModel saved successfully to: {output_path}")
        
        # Save metadata as JSON for easy inspection
        metadata_path = output_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_path}")
    
    def run(self, output_path):
        """Execute the complete training pipeline"""
        print("KNN Classifier Training Pipeline")
        print("=" * 60)
        
        # Load data
        if not self.load_data():
            return False
        
        # Select target column
        if not self.select_target_column():
            return False
        
        # Preprocess data
        X, y = self.preprocess_data()
        
        # Train and evaluate
        X_test, y_test = self.split_and_train(X, y)
        accuracy = self.evaluate_model(X_test, y_test)
        
        # Save model
        self.save_model(output_path)
        
        print("\nTraining pipeline completed successfully!")
        print(f"\nNext steps:")
        print(f"   1. To make predictions: python predict.py {output_path} <new_data.csv>")
        print(f"   2. To view model details: Check {output_path.replace('.pkl', '_metadata.json')}")
        
        return True


def create_parser():
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description='Train a KNN classifier on any CSV dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data.csv                    # Interactive mode
  %(prog)s data.csv -t target_col      # Specify target column
  %(prog)s data.csv -k 7 -w uniform    # Custom parameters
  %(prog)s data.csv -o my_model.pkl    # Custom output path
        """
    )
    
    # Required arguments
    parser.add_argument('input_file', type=str, help='Path to input CSV file')
    
    # Optional arguments
    parser.add_argument('-t', '--target', type=str, help='Target column name (if not specified, will prompt)')
    parser.add_argument('-k', '--neighbors', type=int, default=5, help='Number of neighbors (default: 5)')
    parser.add_argument('-w', '--weights', type=str, default='distance', 
                       choices=['uniform', 'distance'], help='Weight function (default: distance)')
    parser.add_argument('-s', '--test-size', type=float, default=0.2, 
                       help='Test set size as fraction (default: 0.2)')
    parser.add_argument('-r', '--random-state', type=int, default=42, 
                       help='Random state for reproducibility (default: 42)')
    parser.add_argument('-o', '--output', type=str, default='models/knn_model.pkl', 
                       help='Output path for model (default: models/knn_model.pkl)')
    
    return parser


def main():
    """Main execution function"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        sys.exit(1)
    
    if args.test_size <= 0 or args.test_size >= 1:
        print(f"Error: Test size must be between 0 and 1 (got {args.test_size})")
        sys.exit(1)
    
    if args.neighbors < 1:
        print(f"Error: Number of neighbors must be at least 1 (got {args.neighbors})")
        sys.exit(1)
    
    # Create trainer instance
    trainer = KNNTrainer(
        input_file=args.input_file,
        target_column=args.target,
        n_neighbors=args.neighbors,
        weights=args.weights,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # Run training pipeline
    success = trainer.run(args.output)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()