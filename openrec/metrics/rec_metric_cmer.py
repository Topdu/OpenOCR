import re
import math
import collections
from functools import lru_cache
import numpy as np
import datasets
import evaluate
from rouge_score import rouge_scorer, scoring
from Levenshtein import distance as levenshtein_distance


def _get_ngrams(segment, max_order):
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i+order])
            ngram_counts[ngram] += 1
    return ngram_counts

def compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=False):
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus, translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)
        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram)-1] += overlap[ngram]
        for order in range(1, max_order+1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order-1] += possible_matches
    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0
    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    if reference_length == 0:
        ratio = 0.0
    else:
        ratio = float(translation_length) / reference_length
        
    if ratio > 1.0:
        bp = 1.
    elif ratio <= 0: 
        bp = 0.0
    else:
        bp = math.exp(1 - 1. / ratio)
        
    bleu = geo_mean * bp
    return (bleu, precisions, bp, ratio, translation_length, reference_length)

class BaseTokenizer:
    def signature(self):
        return "none"
    def __call__(self, line):
        return line

class TokenizerRegexp(BaseTokenizer):
    def signature(self):
        return "re"
    def __init__(self):
        self._re = [
            (re.compile(r"([\{-\~[-\` -\&\(-\+\:-\@\/])"), r" \1 "),
            (re.compile(r"([^0-9])([\.,])"), r"\1 \2 "),
            (re.compile(r"([\.,])([^0-9])"), r" \1 \2"),
            (re.compile(r"([0-9])(-)"), r"\1 \2 "),
        ]
    @lru_cache(maxsize=2**16)
    def __call__(self, line):
        for (_re, repl) in self._re:
            line = _re.sub(repl, line)
        return line.split()

class Tokenizer13a(BaseTokenizer):
    def signature(self):
        return "13a"
    def __init__(self):
        self._post_tokenizer = TokenizerRegexp()
    @lru_cache(maxsize=2**16)
    def __call__(self, line):
        line = line.replace("<skipped>", "")
        line = line.replace("-\n", "")
        line = line.replace("\n", " ")
        if "&" in line:
            line = line.replace("&quot;", '"')
            line = line.replace("&amp;", "&")
            line = line.replace("&lt;", "<")
            line = line.replace("&gt;", ">")
        return self._post_tokenizer(f" {line} ")

class CustomBleu(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description="Custom BLEU implementation",
            citation="",
            inputs_description="",
            features=datasets.Features({
                "predictions": datasets.Value("string", id="sequence"),
                "references": datasets.Sequence(datasets.Value("string", id="sequence"), id="references"),
            }),
        )
    def _compute(self, predictions, references, tokenizer=None, max_order=4, smooth=False):
        if tokenizer is None:
            tokenizer = Tokenizer13a()
            
        if isinstance(references[0], str):
            references = [[ref] for ref in references]
        references_tokenized = [[tokenizer(r) for r in ref] for ref in references]
        predictions_tokenized = [tokenizer(p) for p in predictions]
        score = compute_bleu(
            reference_corpus=references_tokenized, 
            translation_corpus=predictions_tokenized, 
            max_order=max_order, 
            smooth=smooth
        )
        (bleu, precisions, bp, ratio, translation_length, reference_length) = score
        return {
            "bleu": bleu,
            "precisions": precisions,
            "brevity_penalty": bp,
            "length_ratio": ratio,
            "translation_length": translation_length,
            "reference_length": reference_length,
        }


class CustomRougeTokenizer:
    def __init__(self, tokenizer_func):
        self.tokenizer_func = tokenizer_func
    def tokenize(self, text):
        return self.tokenizer_func(text)

class CustomRouge(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description="Custom ROUGE implementation",
            citation="",
            inputs_description="",
            features=datasets.Features({
                "predictions": datasets.Value("string", id="sequence"),
                "references": datasets.Sequence(datasets.Value("string", id="sequence")),
            }),
        )
    def _compute(
        self, predictions, references, rouge_types=None, use_aggregator=True, use_stemmer=False, tokenizer=None
    ):
        if rouge_types is None:
            rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        
        multi_ref = isinstance(references[0], list)
        

        if tokenizer is not None:
            tokenizer = CustomRougeTokenizer(tokenizer)
            
        scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=use_stemmer, tokenizer=tokenizer)
        
        if use_aggregator:
            aggregator = scoring.BootstrapAggregator()
        else:
            scores = []
            
        for ref, pred in zip(references, predictions):
            if multi_ref:
                score = scorer.score_multi(ref, pred)
            else:
                score = scorer.score(ref, pred)
            if use_aggregator:
                aggregator.add_scores(score)
            else:
                scores.append(score)
                
        if use_aggregator:
            result = aggregator.aggregate()
            for key in result:
                result[key] = result[key].mid.fmeasure
        else:
            result = {}
            first_score = scores[0]
            for key in first_score:
                result[key] = [s[key].fmeasure for s in scores]
        return result


class CMERMetric(object):
    def __init__(self, main_indicator='bleu', **kwargs):
        self.main_indicator = main_indicator

        self.tokenizer = Tokenizer13a()
        self.rouge_metric = CustomRouge()
        self.bleu_metric = CustomBleu()
        self.reset()

    def reset(self):
        self.preds_list = []
        self.labels_list = []

    def _compute_single_pair(self, pred, label):
        preds = [pred]
        refs_formatted = [[label]]


        rouge_results = self.rouge_metric.compute(
            predictions=preds, 
            references=refs_formatted,
            use_aggregator=True,
            tokenizer=self.tokenizer  
        )
        

        bleu_results = self.bleu_metric.compute(
            predictions=preds, 
            references=refs_formatted,
            tokenizer=self.tokenizer 
        )
        
        dist = levenshtein_distance(pred, label)
        
        return {
            "rouge1": rouge_results['rouge1'],
            "rouge2": rouge_results['rouge2'],
            "rougeL": rouge_results['rougeL'],
            "bleu": bleu_results['bleu'],
            "edit_distance": float(dist),
        }

    def __call__(self, preds, labels, **kwargs):
        if isinstance(preds, str):
            preds = [preds]
        if isinstance(labels, str):
            labels = [labels]
        self.preds_list.extend(preds)
        self.labels_list.extend(labels)

    def compute_single(self, preds, labels):
        if len(preds) == 0:
            return {
                "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0,
                "bleu": 0.0, "edit_distance": 0.0, 
            }
        
        total_metrics = collections.defaultdict(float)
        count = 0
        
        for p, l in zip(preds, labels):
            single_res = self._compute_single_pair(p, l)
            for k, v in single_res.items():
                total_metrics[k] += v
            count += 1
            
        return {k: v / count for k, v in total_metrics.items()}

    def get_metric(self):
        if len(self.preds_list) == 0:
            return {
                "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0,
                "bleu": 0.0, "edit_distance": 0.0, 
            }

        total_metrics = collections.defaultdict(float)
        count = len(self.preds_list)

        for p, l in zip(self.preds_list, self.labels_list):
            single_res = self._compute_single_pair(p, l)
            for k, v in single_res.items():
                total_metrics[k] += v

        avg_metrics = {k: v / count for k, v in total_metrics.items()}
        
        self.reset()
        return avg_metrics