from collections import Counter
import re

def extract_keywords(text):
    text = text.lower()
    words = re.findall(r'\b[a-z]{3,}\b', text)  # words with 3+ letters
    
    common = Counter(words).most_common(10)
    return [word for word, count in common]