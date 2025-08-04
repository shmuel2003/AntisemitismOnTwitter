import pandas as pd
from cleaner import DataCleaner
from analyzer import DataAnalyzer

def main():
    # Load dataset
    df = pd.read_csv("../data/tweets_dataset.csv")

    # Clean text
    cleaner = DataCleaner()
    clean_df = cleaner.clean_dataframe(df)

    # Save cleaned data
    clean_df.to_csv("../results/tweets_dataset_cleaned.csv", index=False)

    # Analyze
    analyzer = DataAnalyzer(clean_df)
    analyzer.analyze()
    analyzer.export_results("../results/results.json")

if __name__ == "__main__":
    main()