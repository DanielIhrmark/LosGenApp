"""
Description
Public interface for the LNU LosGen
"""
import sys
import os
from io import StringIO
import openai
from streamlit_chat import message

import nltk
import requests
import streamlit as st
from nltk.tokenize import TreebankWordTokenizer as twt
from nltk.util import ngrams
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from streamlit_lottie import st_lottie
import pandas as pd
import re
from collections import Counter
from string import punctuation

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)


# Loading corpus
losgen = pd.read_pickle("./losgen.pkl")

# Setting up the filter function
def filter_dataframe(keymarker, df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters" + keymarker)

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter corpus" + keymarker, df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}" + keymarker,
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}" + keymarker,
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df


@st.cache_resource
def nltk_download(*args):
	for arg in args:
		nltk.download(arg)

@st.cache_resource
def load_lottieurl(url: str):
	r = requests.get(url)
	if r.status_code != 200:
		return None
	return r.json()

# Function for concordancer TAB0
@st.cache_data
def concordancer(corpus, searchTerm):
	tmp = sys.stdout
	my_result = StringIO()
	sys.stdout = my_result
	searchTermSplit = nltk.word_tokenize(searchTerm)
	tokenizedText = nltk.word_tokenize(corpus)
	concText = nltk.Text(tokenizedText)
	concText.concordance(searchTermSplit, lines= 50)
	sys.stdout = tmp
	return my_result.getvalue()


# Function for frequency lists TAB1
@st.cache_data
def get_freqy(my_text):
	my_text = re.sub(r'[^\w\s]','', my_text.lower())
	tokens = nltk.word_tokenize(my_text)
	stopWords = set(stopwords.words('english'))
	cleanTokens = []
	for t in tokens:
		if t not in stopWords:
			cleanTokens.append(t)

	freqDist = nltk.FreqDist(cleanTokens)
	return freqDist


@st.cache_data
def lemmatizer(my_text):
	my_text = re.sub(r'[^\w\s]','', my_text.lower())
	lmtzr = nltk.WordNetLemmatizer()
	words = nltk.word_tokenize(my_text)
	stopWords = set(stopwords.words('english'))

	lemmatized = []
	
	for w in words:
		if w not in stopWords:
			lemmatized.append(lmtzr.lemmatize(w))

	lemmaFreqDist = nltk.FreqDist(lemmatized)

	return lemmaFreqDist

# Keyword extraction done in place TAB2

# Function for collocations and N-grams TAB3
def ngram_analyzer(my_text, num):
	n_grams = ngrams(nltk.word_tokenize(my_text), num)
	return [ ' '.join(grams) for grams in n_grams]

# Chatbot done in place TAB4

# pre-load nltk packages
nltk_download("punkt", "averaged_perceptron_tagger", "universal_tagset", "stopwords", "wordnet")


def main():
	""" web interface """
	# Set tabs up
	tab0, tab1, tab2, tab3, tab4 = st.tabs(["Current Content", "Concordancer", "Tokens and Lemma", "N-grams", "BETA: LostBot"])


	# Sidebar
	with st.sidebar:
		computer = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_lZfPGMRgUo.json")
		st_lottie(computer)
		st.title("LNU LosGen Web")
		st.subheader("Online interface for the Lost Generation corpus")
		st.subheader("About the App")
		st.info("This is a basic interface intended to help people easily explore the [Lost Generation corpus](https://lnu.se/en/research/research-projects/project-the-lost-generation-corpus/). Each of the tabs contain an interesting way to explore your selection of texts from the current corpus.")
		with open("LNU LostGen Web User Guide.pdf", "rb") as pdf_file:
			PDFbyte = pdf_file.read()
		st.download_button(label="Download Guide",
                    data=PDFbyte,
                    file_name="LNU LostGen Web User Guide.pdf",
                    mime='application/octet-stream')
		st.subheader("Contact and further access to the corpus")
		st.text("Daniel Ihrmark")
		st.text("(daniel.o.sundberg@lnu.se)")

	with tab0:
		st.title('Current content')
		
		losgen2 = losgen.drop('text', axis=1)
		colNames = list(losgen2.columns.values)

		st.dataframe(losgen2)

		userWord = st.text_input("Search", "")
		colChoice = st.selectbox('Select search column:', colNames)
		if st.button("Find"):
			st.dataframe(losgen2.loc[losgen2[colChoice]==userWord], use_container_width=True)

	# Concordancer
	with tab1:
		st.subheader("Concordancer")
		st.info("Concordancing allows us to see a word of our choice in its original context. The resulting lines of text are sometimes referred to as 'Keyword in Context Lines', or KWIC Lines!")
		
		selection = filter_dataframe(".", losgen)
		showSelection = selection.drop('text', axis=1)
		st.dataframe(showSelection)


		query = st.text_area("Enter Query","Type Query.")
		if st.button("Create Concordance"):
			data = selection['text'].str.cat(sep=', ')
			st.text("Concordance lines:")
			st.text(concordancer(data, query))


	# Frequencies
	with tab2:
		st.subheader("Tokens and Lemma")

		st.info("The frequencies of words in our text can tell us a lot about what is going on! We can use a frequency list to look for recurring clues. When we want to talk about the frequencies of words in a text, we often talk about 'Types' and 'Tokens'. Tokens are the individual words of a text, while types are the unique words.")
		st.info("We can also use Lemmatization when we want to know what is going on in a text. Lemmas are the 'dictionary', or base, forms of words. When we 'lemmatize' our text, we collect the base form of each word, which grants our frequency lists a different perspective.")

		selection = filter_dataframe("..", losgen)
		showSelection = selection.drop('text', axis=1)
		st.dataframe(showSelection)

		if st.button("Tokenize"):
			data = selection['text'].str.cat(sep=', ')
			results = get_freqy(data)
			st.dataframe(pd.DataFrame(list(results.items()), columns = ["Token","Frequency"]))

		if st.button("Lemmatize"):
			data = selection['text'].str.cat(sep=', ')
			results = lemmatizer(data)
			st.dataframe(pd.DataFrame(list(results.items()), columns = ["Lemma","Frequency"]))


	#Collocation and N-grams
	with tab3:
		st.subheader("N-grams")
		st.info("'We should know a word by the company it keeps' is one of those quotes every linguistics student will hear from a lecturer at least once during their studies. The reason for this is that the context of a word is very important for formulating our understanding of it. From a corpus linguistics perspective we can look for words that often appear together in text in order to get an idea of the word groups in our specific text.")
		st.info("N-grams are simply groups of words defined by the number of words included. This tool allows us to look for N-grams, and N-grams containing a specific term.")
		
		selection = filter_dataframe("!", losgen)
		showSelection = selection.drop('text', axis=1)
		st.dataframe(showSelection)

		data = selection['text'].str.cat(sep=', ')

		data = re.sub(r'[^\w\s]','', data.lower())

		number = st.number_input("N-gram length", min_value=2)

		ngramWord = st.text_input("Optional: N-grams must include...", "")

		if st.button("Get N-grams"):
			st.text("Using NLTK N-gram Extractor...")
			results = ngram_analyzer(data, number)
			freqResults = nltk.FreqDist(results)
			resultsDF = pd.DataFrame(list(freqResults.items()), columns = ["N-gram","Frequency"])
			resultsDF2 = resultsDF[resultsDF['N-gram'].str.contains(ngramWord)]
			resultsDF3 = resultsDF2.sort_values(by=["Frequency"],ascending=False)
			st.dataframe(resultsDF3.set_index(resultsDF3.columns[0]))
		
	with tab4:
		st.title("LostBot: A LosGen Corpus Helper")
		st.info("This is a helper chatbot that can answer some questions regarding the novels and authors in the Lost Generation corpus. The bot does not have access to the corpus itself, but it can access a lot of outside information regarding authors and their works. It is based on OpenAI's Large Language Model GPT 3.5 Turbo, and it should not be trusted. However, you can ask it questions about the novels and short stories, and then try to verify the answers using the methods available in the interface.")

		understand = st.checkbox('I understand the limitations of LostBot and that I have to verify any statements made by it')

		if understand:
			with open("app_bot.py") as f:
				exec(f.read())
		

if __name__ == '__main__':
	main()

