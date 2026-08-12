# imported from google colab. 
!pip install bertopic sentence-transformers scikit-learn pandas

from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from hdbscan import HDBSCAN
import random
import numpy as np

random.seed(43)
np.random.seed(43)

import nltk
import spacy

from nltk.corpus import stopwords

nltk.download('stopwords')
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

urdu_hindi_stops = [
    "0o", "0s", "2018", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", 
    "ab", "able", "about", "above", "abst", "ac", "accordance", "according", 
    "accordingly", "across", "act", "actually", "ad", "added", "adj", "adnan", 
    "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", 
    "aga", "again", "against", "agha", "agha khan", "ah", "ahh", "ain", "ain't", 
    "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", 
    "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", 
    "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", 
    "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", 
    "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", 
    "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", 
    "ask", "asked", "asking", "associated", "at", "au", "aur", "auth", "av", 
    "available", "average", "aw", "away", "awfully", "ax", "ay", "az", "azam", 
    "b", "b1", "b2", "b3", "ba", "baat", "back", "bc", "bd", "be", "became", 
    "because", "become", "becomes", "becoming", "been", "before", "beforehand", 
    "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", 
    "below", "beside", "besides", "best", "better", "between", "beyond", "bhi", 
    "bht", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "bohat", "bohot", 
    "boht", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", 
    "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", 
    "cannot", "cant", "can't", "cause", "causes", "cavalry", "cc", "cd", "ce", 
    "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", 
    "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", 
    "concerning", "consequently", "consider", "considering", "contain", 
    "containing", "contains", "corresponding", "could", "couldn", "couldnt", 
    "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", 
    "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", 
    "de", "definitely", "describe", "described", "despite", "detail", "df", 
    "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", 
    "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", 
    "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", 
    "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", 
    "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", 
    "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", 
    "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", 
    "every", "everybody", "everyone", "everything", "everywhere", "ex", 
    "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fatima", 
    "fazool", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", 
    "find", "fire", "first", "five", "five star", "fix", "fj", "fl", "fn", 
    "fo", "followed", "following", "follows", "for", "former", "formerly", 
    "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", 
    "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", 
    "get", "gets", "getting", "gi", "give", "given", "gives", "giving", 
    "gj", "gl", "go", "goes", "going", "gone", "google", "got", "gotten", 
    "govt", "government", "gr", "greetings", "gs", "gy", "gya", "gye", "h", 
    "h2", "h3", "had", "hadn", "hadn't", "ha", "hai", "hain", "han", "happens", 
    "hardly", "has", "hasn", "hasnt", "hasn't", "hate", "have", "haven", 
    "haven't", "having", "hay", "he", "hed", "he'd", "he'll", "hello", "help", 
    "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", 
    "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", 
    "himself", "his", "hither", "hj", "hn", "ho", "hoga", "hogi", "home", 
    "hopefully", "hospital", "hota", "how", "howbeit", "however", "how's", 
    "hr", "hs", "http", "hu", "hum", "hundred", "hy", "i", "i2", "i3", "i4", 
    "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", 
    "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", 
    "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", 
    "index", "indicate", "indicated", "indicates", "indus", "information", 
    "inner", "insofar", "instead", "interest", "international", "into", "invention", 
    "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", 
    "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jinna", 
    "jinnah", "jj", "jo", "jr", "js", "jt", "ju", "just", "k", "ka", "kaha", 
    "kar", "karna", "ke", "keep", "keeps", "kept", "kg", "khan", "khanum", 
    "kia", "kisi", "kiun", "ki", "kj", "km", "know", "known", "knows", "ko", 
    "koi", "kr", "kuch", "ky", "kya", "kyun", "l", "l2", "la", "largely", 
    "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", 
    "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", 
    "likely", "line", "little", "liye", "lj", "ll", "ln", "lo", "log", "logo", 
    "logon", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", 
    "m2", "ma", "made", "mai", "main", "mainly", "make", "makes", "many", 
    "maps", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", 
    "medicine", "medicines", "mein", "mera", "merely", "meri", "mg", "might", 
    "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", 
    "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", 
    "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", 
    "na", "nahi", "nahin", "nai", "name", "namely", "national", "nay", "nc", 
    "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", 
    "needn't", "needs", "neither", "never", "nevertheless", "new", "next", 
    "ng", "nhi", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", 
    "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", 
    "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", 
    "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", 
    "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", 
    "omitted", "on", "once", "one", "one star", "ones", "only", "onto", "oo", 
    "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", 
    "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", 
    "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", 
    "pages", "par", "part", "particular", "particularly", "pas", "past", "pathetic", 
    "patient", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "phir", "pi", 
    "pim", "pims", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", 
    "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", 
    "predominantly", "present", "presumably", "previously", "primarily", 
    "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", 
    "py", "q", "qj", "qu", "quaid", "que", "quickly", "quite", "qv", "r", "r2", 
    "ra", "raha", "rahi", "rahe", "ran", "rather", "rc", "rd", "re", "readily", 
    "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", 
    "regardless", "regards", "related", "relatively", "research", "research-articl", 
    "respectively", "resulted", "resulting", "results", "rf", "rh", "rha", "ri", 
    "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", 
    "rv", "ry", "s", "s2", "sa", "sab", "said", "same", "saw", "say", "saying", 
    "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", 
    "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", 
    "sensible", "sent", "serious", "seriously", "seven", "several", "sf", 
    "shall", "shan", "shan't", "shamin", "she", "shed", "she'd", "she'll", 
    "shes", "she's", "shifa", "should", "shouldn", "shouldn't", "should've", 
    "show", "showed", "shown", "showns", "shows", "shukriya", "si", "side", 
    "significant", "significantly", "similar", "similarly", "since", "sincere", 
    "sirf", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", 
    "somebody", "somehow", "someone", "somethan", "something", "sometime", 
    "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", 
    "specified", "specify", "specifying", "sq", "sr", "ss", "st", "star", 
    "stars", "still", "stop", "strongly", "sub", "substantially", "successfully", 
    "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", 
    "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", 
    "tell", "ten", "tends", "tf", "th", "tha", "than", "thank", "thanks", 
    "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", 
    "theirs", "them", "themselves", "then", "thence", "there", "thereafter", 
    "thereby", "thered", "therefore", "therein", "there'll", "thereof", 
    "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", 
    "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", 
    "thickv", "thi", "thin", "think", "third", "this", "thorough", "thoroughly", 
    "those", "thou", "though", "thoughh", "thousand", "three", "throug", 
    "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", 
    "tm", "tn", "to", "together", "told", "too", "took", "top", "toward", 
    "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", 
    "ts", "t's", "tt", "tu", "tv", "twelve", "twenty", "twice", "two", "tx", 
    "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", 
    "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", 
    "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", 
    "using", "usually", "ut", "v", "va", "value", "van", "various", "vd", "ve", 
    "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", 
    "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", 
    "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", 
    "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", 
    "what'll", "whats", "what's", "when", "whence", "whenever", "when's", 
    "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", 
    "whereupon", "wherever", "whether", "which", "while", "whim", "whither", 
    "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", 
    "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", 
    "with", "within", "without", "wo", "wo", "won", "wonder", "wont", "won't", 
    "words", "world", "worst", "would", "wouldn", "wouldnt", "wouldn't", "www", 
    "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", 
    "xv", "xx", "ya", "ye", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", 
    "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", 
    "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz"
]

hospital_stops = [
    "1977", "admited", "aga", "allah", "anonymously", "anymore", "apna", 
    "appointment", "assistant", "bad", "beautiful", "brother", "buht", "care", 
    "caring", "center", "clinic", "dad", "daughter", "death", "discharged", 
    "doctor", "doctors", "dont", "dr", "emergancy", "even", "excellent", 
    "experience", "extremely", "facility", "father", "fathers", "family", 
    "ghurki", "good", "great", "gud", "hamid", "health", "healthcare", "helpful", 
    "hes", "horrible", "hospital", "hospitals", "hotel", "islamabad", "karachi", 
    "khan", "know", "lahore", "latif", "life", "like", "loved", "medical", 
    "mera", "mother", "nai", "nero", "nice", "nurse", "nurses", "one", "okay", 
    "pakistan", "pata", "patient", "patients", "peshawar", "physician", "practitioner", 
    "private", "quetta", "receptionist", "recommend", "rmi", "rn", "satisfied", 
    "sath", "sb", "service", "services", "shaukat", "sister", "sisters", 
    "specialist", "staff", "surgeon", "terrible", "treatment", "unit", "us", 
    "vip", "visit", "wala", "ward", "wonderful"
]

eng_stops = list(stopwords.words('english'))

final_stopwords = eng_stops + urdu_hindi_stops + hospital_stops

from bertopic import BERTopic
import pandas as pd

# Loading dataset here
df = pd.read_csv('low_rating_lahore_reviews.csv')
docs = df['cleaned_text'].astype(str).tolist()

final_stopwords = set([word.lower() for word in (eng_stops + urdu_hindi_stops + hospital_stops)])

def strict_clean_for_bertopic(texts):
    cleaned = []
    for doc in nlp.pipe(texts, batch_size=50):
        tokens = []
        for t in doc:
            if t.pos_ in ["NOUN", "ADJ", "PROPN"]:
                lemma = t.lemma_.lower().strip()
                if lemma not in final_stopwords and len(lemma) > 2:
                    tokens.append(lemma)
        cleaned.append(" ".join(tokens))
    return cleaned

from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired
from hdbscan import HDBSCAN

cleaned_docs = strict_clean_for_bertopic(docs)

vectorizer_model = CountVectorizer(stop_words=list(final_stopwords), ngram_range=(1, 2))

from umap import UMAP

umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42,
    transform_seed=42
)

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    "paraphrase-multilingual-mpnet-base-v2"
)

hdbscan_model = HDBSCAN(
    min_cluster_size=10,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)

representation_model = {
    "KeyBERT": KeyBERTInspired(),
    "MMR": MaximalMarginalRelevance(diversity=0.4)
}

topic_model_4 = BERTopic(
    umap_model=umap_model,
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,
    hdbscan_model=hdbscan_model,
    representation_model=representation_model,
    verbose=True
)

topics, probs = topic_model_4.fit_transform(cleaned_docs)

new_topics = topic_model_4.reduce_outliers(cleaned_docs, topics, strategy="embeddings")
topic_model_4.update_topics(docs, topics=new_topics, vectorizer_model=vectorizer_model)

topic_info = topic_model_4.get_topic_info()
print(topic_info.head(100))

print((topic_model_4.get_representative_docs(27)))

print("Original outliers:", sum(np.array(topics) == -1))
print("Remaining outliers:", sum(np.array(new_topics) == -1))

topic_model_4.visualize_hierarchy()

topic_info = topic_model_4.get_topic_info()

# Save only Topic ID and Name
topic_info[["Topic", "Name"]].to_csv(
    "topic_names_FINAL.csv",
    index=False
)

df['theme'] = new_topics

df.to_csv('low_rating_lahore_reviews_with_topics.csv', index=False)

df_mapping = pd.read_csv("topic_names_with_themes.csv")

df = df.merge(
    df_mapping[['Topic', 'Theme', 'Sub_Theme']],
    left_on='theme',
    right_on='Topic',
    how='left'
)

df = df.drop(columns=['Topic'])

df_tot = pd.read_csv('cleaned_lahore_reviews.csv')

"""Processing Missed Data"""

df_tot_new = pd.read_csv('cleaned_missed_hospitals.csv')

df_new = pd.read_csv('low_rating_missed_hospital_reviews.csv')

source_column = None
if 'cleaned_text' in df_new.columns:
    source_column = 'cleaned_text'
    print("Using 'cleaned_text' column from df_new as source for strict cleaning.")
elif 'text' in df_new.columns:
    source_column = 'text'
    print("Using 'text' column from df_new as source for strict cleaning.")
else:
    raise ValueError("Neither 'cleaned_text' nor 'text' column found in df_new. Cannot proceed with cleaning.")

new_cleaned_docs = strict_clean_for_bertopic(df_new[source_column].astype(str).tolist())

new_topics_predicted, _ = topic_model_4.transform(new_cleaned_docs)

df_new['theme'] = new_topics_predicted

df_new_merged = df_new.merge(
    df_mapping[['Topic', 'Theme', 'Sub_Theme']],
    left_on='theme',
    right_on='Topic',
    how='left'
)

df_new_merged = df_new_merged.drop(columns=['Topic'])

all_new_hospitals = df_new_merged['title'].dropna().unique()

all_major_themes = df_mapping['Theme'].dropna().unique()

structured_summary_new = []

for hospital in all_new_hospitals:
    hospital_total_reviews = df_tot_new[df_tot_new['title'] == hospital]
    hospital_neg_reviews = df_new_merged[df_new_merged['title'] == hospital]

    count_total = len(hospital_total_reviews)
    count_negative = len(hospital_neg_reviews)
    pct_negative = (count_negative / count_total * 100) if count_total > 0 else 0.0

    process_specific_df = hospital_neg_reviews[
        hospital_neg_reviews['Theme'] == "process specific"
    ]

    major_distributions = hospital_neg_reviews['Theme'].value_counts(normalize=True) * 100

    sub_distributions = (
        process_specific_df['Sub_Theme'].value_counts(normalize=True) * 100
        if len(process_specific_df) > 0 else pd.Series()
    )

    hospital_row = {
        "Hospital Name": hospital,
        "Total Reviews (All Stars)": count_total,
        "Negative Reviews Count (1-3 Stars)": count_negative,
        "Negative Reviews Ratio (%)": round(pct_negative, 2)
    }

    for theme in all_major_themes:
        hospital_row[f"Major Theme: {theme} (%)"] = round(major_distributions.get(theme, 0.0), 2)

    all_sub_themes = df_mapping[df_mapping['Theme'] == "process specific"]['Sub_Theme'].dropna().unique()

    for sub_theme in all_sub_themes:
        hospital_row[f"Sub-Theme: {sub_theme} (%)"] = round(sub_distributions.get(sub_theme, 0.0), 2)

    for theme in all_major_themes:
        docs_slice = hospital_neg_reviews[
            hospital_neg_reviews['Theme'] == theme
        ]['cleaned_text'].head(2).tolist()

        cleaned_docs_rep = [doc.replace('\n', ' ').strip() for doc in docs_slice]
        hospital_row[f"Representative Docs (Major): {theme}"] = (
            " || ".join(cleaned_docs_rep) if cleaned_docs_rep else "No reviews available"
        )

    for sub_theme in all_sub_themes:
        docs_slice = process_specific_df[
            process_specific_df['Sub_Theme'] == sub_theme
        ]['cleaned_text'].head(2).tolist()

        cleaned_docs_rep = [doc.replace('\n', ' ').strip() for doc in docs_slice]
        hospital_row[f"Representative Docs (Sub): {sub_theme}"] = (
            " || ".join(cleaned_docs_rep) if cleaned_docs_rep else "No reviews available"
        )

    structured_summary_new.append(hospital_row)

df_export_new = pd.DataFrame(structured_summary_new)
df_export_new = df_export_new.sort_values(by="Hospital Name")

print("Metrics for new hospitals calculated.")

final_data_old = pd.read_csv('final_data.csv')

for col in final_data_old.columns:
    if col not in df_export_new.columns:
        if ' (%)' in col:
            df_export_new[col] = 0.0
        elif 'Representative Docs' in col:
            df_export_new[col] = 'No reviews available'
        else:
            df_export_new[col] = None

df_export_new = df_export_new[final_data_old.columns]

final_data_updated = pd.concat([final_data_old, df_export_new], ignore_index=True)

final_data_updated.to_csv('final_data.csv', index=False)

df_export.columns
