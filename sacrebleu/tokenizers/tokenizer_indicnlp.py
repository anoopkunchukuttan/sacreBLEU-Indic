from functools import lru_cache

from .tokenizer_base import BaseTokenizer

try:
    import indicnlp
    from indicnlp import loader
    loader.load()    
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    from indicnlp.tokenize import indic_tokenize 
except ImportError:
    # Don't fail until the tokenizer is actually used
    indicnlp = None



FAIL_MESSAGE = """
Indic tokenization requires extra dependencies, but you do not have them installed.
Please install them like so.

    pip install sacrebleu[indic]
"""

class TokenizerIndicNLP(BaseTokenizer)::
    """Indic language tokenizer."""

    def __init__(lang):
        if indicnlp is None:
            raise RuntimeError(FAIL_MESSAGE)        
        factory=IndicNormalizerFactory()
        self.normalizer=factory.get_normalizer(lang)  

    def signature(self):
        """
        Returns a signature for the tokenizer.

        :return: signature string
        """
        return 'indicnlp-' + indicnlp.__version__

    def __call__(self, line):
        """
        Tokenizes an input line with the tokenizer.

        :param line: a segment to tokenize
        :return: the tokenized line
        """
        return ' '.join(indic_tokenize.trivial_tokenize(
                        self.normalizer.normalize(line) ))