#عمار ابوسنينة   23009021
import re

USE_STOPWORDS = True      
USE_STEMMING = True       

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "by",
    "for", "with", "from", "as", "is", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "which", "who", "he",
    "she", "they", "his", "her", "their", "also", "has", "have", "had",
    "not", "than", "then", "there", "into", "after", "before", "about",
}

def tokenize(text):
   
    return re.findall(r"[a-z0-9]+", text.lower())

def remove_stopwords(tokens):
    
    return [t for t in tokens if t not in STOPWORDS]

def stem_word(word):
    for suffix in ["ing", "ed", "es", "s"]:
        if word.endswith(suffix) and not word.endswith("ss") \
                and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word

def preprocess(text, use_stopwords=USE_STOPWORDS, use_stemming=USE_STEMMING):
    terms = tokenize(text)
    if use_stopwords:
        terms = remove_stopwords(terms)
    if use_stemming:
        terms = [stem_word(t) for t in terms]
    return terms

def show_before_after(doc):
    full_text = doc["title"] + " " + doc["text"]
    print("Document:", doc["id"], "-", doc["title"])
    print("1) Original text :", full_text[:200], "...")
    print("2) Tokens        :", tokenize(full_text)[:15], "...")
    print("3) Final terms   :", doc["terms"][:15], "...")
