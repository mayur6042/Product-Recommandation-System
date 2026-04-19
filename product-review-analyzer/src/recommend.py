def recommend(sentiments):
    positive = sentiments.count("positive")
    negative = sentiments.count("negative")

    if positive > negative:
        return "✅ Recommended"
    else:
        return "❌ Not Recommended"