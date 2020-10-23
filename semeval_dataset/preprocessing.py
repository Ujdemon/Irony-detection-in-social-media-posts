import nltk
from tweet_utils import *

nltk_tokeniser = nltk.tokenize.TweetTokenizer()


def tweet_tokenizer(text):
    return simpleTokenize(squeezeWhitespace(text))


def tokeniser(text, with_process=True):
    if with_process:
        return nltk_tokeniser.tokenize(tweet_processor(text).lower())
    else:
        # return nltk_tokeniser.tokenize(text)
        return tweet_tokenizer(text.lower())


def tweet_processor(text):
    FLAGS = re.MULTILINE | re.DOTALL

    def megasplit(pattern, string):
        splits = list((m.start(), m.end()) for m in re.finditer(pattern, string))
        starts = [0] + [i[1] for i in splits]
        ends = [i[0] for i in splits] + [len(string)]
        return [string[start:end] for start, end in zip(starts, ends)]

    def hashtag(text):
        text = text.group()
        hashtag_body = text[1:]
        # print(hashtag_body)

        # result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
        result = " ".join(["<hashtag>"] + megasplit(r"(?=[A-Z])", hashtag_body))
        return result

    def allcaps(text):
        text = text.group()
        return text.lower() + " <allcaps>"

    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"/", " / ")
    text = re_sub(r"@\w+", "<user>")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
    text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
    text = re_sub(r"<3", "<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"#\S+", hashtag)
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")

    ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    text = re_sub(r"([A-Z]){2,}", allcaps)

    return text

