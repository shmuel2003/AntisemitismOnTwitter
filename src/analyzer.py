import pandas as pd
import json
from collections import Counter

class DataAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = {}

    def count_tweets(self):
        counts = self.df['biased'].value_counts().to_dict()
        total = len(self.df)
        unspecified = total - (counts.get(0, 0) + counts.get(1, 0))
        self.results["total_tweets"] = {
            "antisemitic": counts.get(1, 0),
            "non_antisemitic": counts.get(0, 0),
            "total": total,
            "<unspecified>": unspecified
        }

    def average_length(self):
        self.df["length"] = self.df["text"].apply(lambda x: len(x.split()))
        avg = self.df.groupby("biased")["length"].mean().to_dict()
        total_avg = self.df["length"].mean()
        self.results["average_length"] = {
            "antisemitic": round(avg.get(1, 0), 2),
            "non_antisemitic": round(avg.get(0, 0), 2),
            "total": round(total_avg, 2)
        }

    def top_longest_tweets(self):
        self.df["length"] = self.df["text"].apply(lambda x: len(x.split()))
        top_n = lambda group: group.sort_values("length", ascending=False)["text"].head(3).tolist()
        groups = self.df.groupby("biased")
        self.results["longest_3_tweets"] = {
            "antisemitic": top_n(groups.get_group(1)) if 1 in groups.groups else [],
            "non_antisemitic": top_n(groups.get_group(0)) if 0 in groups.groups else []
        }

    def most_common_words(self, n=10):
        all_words = ' '.join(self.df["text"]).split()
        most_common = [word for word, _ in Counter(all_words).most_common(n)]
        self.results["common_words"] = {"total": most_common}

    def uppercase_words_count(self):
        def count_uppercase(text):
            return sum(1 for word in text.split() if word.isupper())

        grouped = self.df.groupby("biased")["text"].apply(
            lambda texts: sum(count_uppercase(t) for t in texts)
        ).to_dict()
        total = sum(grouped.values())
        self.results["uppercase_words"] = {
            "antisemitic": grouped.get(1, 0),
            "non_antisemitic": grouped.get(0, 0),
            "total": total
        }

    def analyze(self):
        self.count_tweets()
        self.average_length()
        self.top_longest_tweets()
        self.most_common_words()
        self.uppercase_words_count()
        return self.results

    def export_results(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4, ensure_ascii=False)