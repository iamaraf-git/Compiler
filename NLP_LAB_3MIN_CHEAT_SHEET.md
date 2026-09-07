# ⚡ 3-Minute NLP Lab Test Cheat Sheet
### Course: NLP / Compiler Construction Lab (`NLP_lab_Spring_26.ipynb`)

---

## 📌 1. Essential Imports & NLTK Downloads

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from collections import defaultdict, OrderedDict

# Download necessary NLTK datasets (run once if needed)
nltk.download('stopwords')
nltk.download('punkt')
```

---

## 🧹 LAB 1: Text Cleaning Pipeline

```python
def clean_text(text):
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove all numbers/digits (\d = 0-9)
    text = re.sub(r'\d', '', text)
    
    # 3. Remove punctuation ([^\w\s] = NOT word & NOT space)
    text = re.sub(r'[^\w\s]', '', text)
    
    # 4. Remove leading and trailing spaces
    text = text.strip()
    
    # 5. Remove stopwords
    sw = set(stopwords.words('english'))
    words = [w for w in text.split() if w not in sw]
    
    return " ".join(words)

# Example:
raw = "   Welcome,, to Compiler lab '55' and '6' course! Here you can learn NLP.   "
print(clean_text(raw))
# Output: "welcome compiler lab course learn nlp"
```

---

## 🔍 LAB 2: Regex Extraction & Custom Frequency Counter

### 1. Pattern Extraction
```python
text = "The Gmail is compiler@gmail.com and nlp@gmail.com. Lab 1703, Room 1401."

# Emails (\S+ = 1 or more non-whitespace characters)
emails = re.findall(r'\S+@\S+', text)
# Output: ['compiler@gmail.com', 'nlp@gmail.com.']

# Numbers / Digits ([0-9]+ or \d+)
numbers = re.findall(r'[0-9]+', text)
# Output: ['1703', '1401']

# Words only ([a-zA-Z]+)
words = re.findall(r'[a-zA-Z]+', text)
```

### 2. Custom Tokenizer with `defaultdict` & `OrderedDict`
```python
def token_counter(text):
    low = text.lower()
    
    # \w+ matches words; [^\w\s] matches punctuation tokens
    tokens = re.findall(r'\w+|[^\w\s]', low)
    
    # Count frequency safely without KeyError
    freq = defaultdict(int)
    for t in tokens:
        freq[t] += 1
        
    # Maintain appearance order without duplicates
    order_freq = OrderedDict()
    for t in tokens:
        if t not in order_freq:
            order_freq[t] = freq[t]
            
    return tokens, order_freq
```

---

## 📚 LAB 3: NLTK Tokenization & Bengali NLP

### 1. NLTK Word & Sentence Tokenization
```python
text = "Welcome to Compiler lab course. Here you can learn NLP."

# Word Tokenization (splits words and punctuation marks)
words = word_tokenize(text)
# Output: ['Welcome', 'to', 'Compiler', 'lab', 'course', '.', 'Here', 'you', 'can', 'learn', 'NLP', '.']

# Sentence Tokenization (splits complete sentences)
sentences = sent_tokenize(text)
# Output: ['Welcome to Compiler lab course.', 'Here you can learn NLP.']

# Frequency Distribution with FreqDist
freq = FreqDist(word_tokenize(text.lower()))
for word, count in freq.items():
    print(f"{word}: {count}")
```

### 2. Bengali NLP (`indic-nlp-library`)
```python
# pip install indic-nlp-library
from indicnlp.tokenize import indic_tokenize, sentence_tokenize

bsent = "আমি বাংলা তে লিখছি । তুমিও কি বাংলা তে লিখছো?"

# Bengali Word Tokenization
b_words = indic_tokenize.trivial_tokenize(bsent)
# Output: ['আমি', 'বাংলা', 'তে', 'লিখছি', '।', 'তুমিও', 'কি', 'বাংলা', 'তে', 'লিখছো', '?']

# Bengali Sentence Tokenization (lang='bn')
b_sents = sentence_tokenize.sentence_split(bsent, lang='bn')
# Output: ['আমি বাংলা তে লিখছি ।', 'তুমিও কি বাংলা তে লিখছো?']
```

---

## 🧠 Regex Quick Memory Card

| Symbol | Meaning | Example Match |
| :--- | :--- | :--- |
| `\d` | Any digit `0-9` | `'5'` |
| `\D` | Any **non**-digit | `'a'`, `'!'` |
| `\w` | Any word char (`a-z`, `A-Z`, `0-9`, `_`) | `'word'`, `'lab1'` |
| `\W` | Any **non**-word char | `','`, `'.'`, `'@'` |
| `\s` | Any whitespace (space, tab, newline) | `' '`, `'\n'` |
| `\S` | Any **non**-whitespace | Letters, symbols |
| `+` | One or more repetitions | `\d+` $\rightarrow$ `'1703'` |
| `[^...]` | **NOT** whatever is inside brackets | `[^\w\s]` $\rightarrow$ punctuation |
| `\S+@\S+` | Email address pattern | `'test@mail.com'` |

---

## 💡 Quick Viva / Teacher Q&A

1. **Q: Why use `defaultdict(int)` instead of `dict`?**  
   *A: To prevent `KeyError` when counting a word for the first time; it automatically initializes missing keys to `0`.*
2. **Q: Why does standard NLTK fail on Bengali sentences?**  
   *A: NLTK splits sentences on English periods (`.`), not Bengali Dari (`।`). We use `indic-nlp-library` with `lang='bn'` instead.*
3. **Q: What is the difference between `word_tokenize()` and `sent_tokenize()`?**  
   *A: `word_tokenize()` splits text into individual word and punctuation tokens; `sent_tokenize()` splits text into full sentences.*
