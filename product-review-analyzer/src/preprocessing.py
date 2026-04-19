from src.predict import predict_sentiment

def extract_pros_cons(reviews):
    pros = []
    cons = []

    for review in reviews:
        sentiment = predict_sentiment(review)

        if sentiment == "positive":
            pros.append(review)
        else:
            cons.append(review)

    return pros, cons