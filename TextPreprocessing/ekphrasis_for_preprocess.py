from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from TextPreprocessing.TextPreprocessor import TextPreprocessor


class EkphrasisTextPreprocess(TextPreprocessor):
    def __init__(self):
        super(EkphrasisTextPreprocess, self).__init__(name='ekphrasis')
        self.tool = get_text_processor()

    def pre_process_doc(self, doc: str):
        return self.tool.pre_process_doc(doc)


def get_text_processor()->TextPreProcessor:
    text_processor = TextPreProcessor(
        # terms that will be normalized
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                   'time', 'url', 'date', 'number'],
        # terms that will be annotated
        annotate={"hashtag", "allcaps", "elongated", "repeated",
                  'emphasis', 'censored'},
        fix_html=True,  # fix HTML tokens

        # corpus from which the word statistics are going to be used
        # for word segmentation
        segmenter="twitter",

        # corpus from which the word statistics are going to be used
        # for spell correction
        corrector="twitter",

        unpack_hashtags=True,  # perform word segmentation on hashtags
        unpack_contractions=True,  # Unpack contractions (can't -> can not)
        spell_correct_elong=False,  # spell correction for elongated words

        # select a tokenizer. You can use SocialTokenizer, or pass your own
        # the tokenizer, should take as input a string and return a list of tokens
        tokenizer=SocialTokenizer(lowercase=True).tokenize,

        # list of dictionaries, for replacing tokens extracted from the text,
        # with other expressions. You can pass more than one dictionaries.
        dicts=[emoticons]
    )
    return text_processor


def test():
    text_processor=get_text_processor()
    sentences = [
        "CANT WAIT for the new season of #TwinPeaks ＼(^o^)／!!! #davidlynch #tvseries :)))",
        "I saw the new #johndoe movie and it suuuuucks!!! WAISTED $10... #badmovies :/",
        "@SentimentSymp:  can't wait for the Nov 9 #Sentiment talks!  YAAAAAAY !!! :-D http://sentimentsymposium.com/."
    ]

    for s in sentences:
        print(" ".join(text_processor.pre_process_doc(s)))
