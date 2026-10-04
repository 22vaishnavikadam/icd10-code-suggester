from icd_suggester import IcdSuggester

m = IcdSuggester().fit()
m.save("model.joblib")
print("Saved model.joblib | metrics:", m.metrics)
