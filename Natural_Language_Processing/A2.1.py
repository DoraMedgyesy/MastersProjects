import nltk
from nltk.corpus import brown
from collections import Counter
import re

# Download if not already done
nltk.download('brown')
nltk.download('punkt')

def tokenize_and_clean(words):
    # Convert to lowercase and keep only alphabetic words
    return [word.lower() for word in words if re.match("^[a-zA-Z]+$", word)]

# All words in the corpus
all_words = brown.words()
clean_all_words = tokenize_and_clean(all_words)

# Frequency distribution
freq_dist_all = Counter(clean_all_words)
sorted_all = freq_dist_all.most_common()

print(sorted_all)