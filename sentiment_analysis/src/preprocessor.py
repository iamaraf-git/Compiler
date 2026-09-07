"""
Text Preprocessor Module for Sentiment Analysis
"""
import re
import string

# Built-in minimal English stopwords to ensure offline resilience
DEFAULT_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
    'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'
}

def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Cleans raw text by:
    1. Removing HTML tags
    2. Removing URLs
    3. Removing punctuation and non-alphabet characters
    4. Converting to lowercase
    5. Normalizing whitespace
    6. Optionally filtering common stopwords
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML tags (common in IMDb dataset e.g. <br />)
    text = re.sub(r'<.*?>', ' ', text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation & numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Normalize whitespace
    words = text.split()

    if remove_stopwords:
        words = [w for w in words if w not in DEFAULT_STOPWORDS]

    return " ".join(words)

if __name__ == "__main__":
    sample = "<br />This movie was simply AMAZING!!! Check it out at https://movie.com <br />"
    print("Original:", sample)
    print("Cleaned: ", clean_text(sample))
