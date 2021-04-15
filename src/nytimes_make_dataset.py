import re
import time
import unicodedata
import requests

from segtok.tokenizer import split_contractions, word_tokenizer


strange_double_quotes = [
    "«",
    "‹",
    "»",
    "›",
    "„",
    "“",
    "‟",
    "”",
    "❝",
    "❞",
    "❮",
    "❯",
    "〝",
    "〞",
    "〟",
    "＂",
]
strange_single_quotes = ["‘", "‛", "’", "❛", "❜", "`", "´", "‘", "’"]

DOUBLE_QUOTE_REGEX = re.compile("|".join(strange_double_quotes))
SINGLE_QUOTE_REGEX = re.compile("|".join(strange_single_quotes))


def fix_strange_quotes(text):
    """
    Replace strange quotes, i.e., 〞with a single quote ' or a double quote " if it fits better.
    """
    text = SINGLE_QUOTE_REGEX.sub("'", text)
    text = DOUBLE_QUOTE_REGEX.sub('"', text)
    return text


def preprocess_text(text):
    """
    Basic preprocessing routine to normalize characters & whitespace and enable splits at " "

    Inputs:
        text: original text
    Returns:
        text: preprocessed text
    """
    # remove html tags (mostly links)
    text = re.sub(r"<.*?>", "", text)
    # replace strange quotes
    text = fix_strange_quotes(text)
    # fix all other no non-ascii characters
    nfkd_form = unicodedata.normalize("NFKD", text)
    text = nfkd_form.encode("ASCII", "ignore").decode("ASCII")
    # normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # whitespace-tokenize how the language models like it:
    # get tokens & concatenate all with whitespace
    if text.strip():
        text = " ".join(split_contractions(word_tokenizer(text)))
    return text.strip()


def download_nytimes_archive(year, month):
    """
    Download articles from NYTimes for the given year and month

    Inputs:
        year, month: integers indicating the year and month for which to download articles
    Returns:
        texts: a list of articles
    """
    # request an API key for the NYTimes Archive API: https://developer.nytimes.com/
    # and save it in a file called 'nytimes_apikey.txt'
    try:
        with open('nytimes_apikey.txt') as f:
            api_key = f.read().strip()
    except:
        raise RuntimeError("Please request an API key for the NYTimes Archive API from https://developer.nytimes.com/ "
                           "and save it in a file called 'nytimes_apikey.txt'")
    # download articles for given month and year
    url = f"https://api.nytimes.com/svc/archive/v1/{year}/{month}.json"
    response = requests.get(url, params={"api-key": api_key}).json()['response']
    assert len(response['docs']) == response['meta']['hits'], "did not receive all articles..."
    texts = []
    for i, article in enumerate(response['docs']):
        snippet, abstract = article.get("snippet", ""), article.get("abstract", "")
        # preprocess text
        abstract = preprocess_text(max([snippet, abstract], key=lambda x: len(x)).replace("To the Editor:", " "))
        headline = preprocess_text(article['headline']['main'].replace("To the Editor:", " ").replace("Corrections", "", 1))
        if headline and headline[-1].isalnum():
            # the headline should end with a punctuation mark to ensure it counts as a different sentence
            headline = headline + " ."
        article_text = f"{headline} {abstract}"
        # filter out some of the less informative articles
        if len(article_text) > 50:
            texts.append(f"{article['pub_date'].split('T')[0]}\t{article_text.strip()}\n")
    return texts


def get_articles(date_begin="2019-01", date_end="2020-12", fname="../data/nytimes_dataset.txt"):
    """
    Download monthly articles from NYTimes between date_begin and date_end (both inclusive) and save in file.
    Please note that the date given for some NYTimes article snippets is noisy, because an article can be updated
    again at a much later date without this being reflected in the data obtained from the API.

    Inputs:
        date_begin, date_end: strings with dates in the format "%YYYY-%MM", e.g. '2017-01'
    """
    year_begin, month_begin = int(date_begin.split('-')[0]), int(date_begin.split('-')[1])
    year_end, month_end = int(date_end.split('-')[0]), int(date_end.split('-')[1])
    # get articles from month_begin/year_begin until month_end/year_end
    texts = []
    year, month = year_begin, month_begin
    while (year < year_end) or (year == year_end and month <= month_end):
        print(f"downloading {year}-{month:02}")
        texts = download_nytimes_archive(year, month)
        with open(fname, "a") as f:
            f.writelines(texts)
        if month < 12:
            month += 1
        else:
            month = 1
            year += 1
        time.sleep(5)


if __name__ == '__main__':
    get_articles()
