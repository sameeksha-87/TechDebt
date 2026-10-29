from src.data_loader import load_dataset
from src.preprocess import preprocess_data
from src.train import train_and_select_best_model
from src.evaluate import evaluate_feature_importances
from src.predict import predict_file_risk

def main():
    print("=== Starting Technical Debt Prediction Pipeline ===")
    
    # 1. Load Data
    df = load_dataset(use_cache=True)

    # 2. Preprocess & Scale
    X_train, X_test, y_train, y_test, feature_cols = preprocess_data(df)

    # 3. Model Training & Selection
    best_model, _ = train_and_select_best_model(X_train, X_test, y_train, y_test)

    # 4. Feature Importance Analysis
    evaluate_feature_importances(best_model, feature_cols)

    # 5. Inference Demonstration
    print("\n--- Running Risk Score Inference Sample ---")
    scored_sample = predict_file_risk(df.iloc[:10].copy(), feature_cols)
    print(scored_sample[['Risk_Score', 'Risk_Category']])

if __name__ == "__main__":
    main()