#!/usr/bin/env python

import json
import codecs

SENTENCE_PAIR_DATA = True
FIXED_VOCABULARY = None
import numpy as np
LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    # Used in the unlabeled test set---needs to map to some arbitrary label.
    "hidden": 0,
}


def convert_binary_bracketing(parse, lowercase=False):
    transitions = []
    tokens = []

    for word in parse.split(' '):
        if word[0] != "(":
            if word == ")":
                transitions.append(1)
            else:
                # Downcase all words to match GloVe.
                if lowercase:
                    tokens.append(word.lower())
                else:
                    tokens.append(word)
                transitions.append(0)
    return tokens, transitions

def get_length_average(s1, s2):
    return float(len(s1.split(" "))+len(s2.split(" ")))/2

def load_data(path, lowercase=False, choose=lambda x: True, eval_mode=False, level="all"):
    print "Loading", path
    examples = []
    failed_parse = 0
    with codecs.open(path, encoding='utf-8') as f:
        for line in f:
            try:
                line = line.encode('UTF-8')
            except UnicodeError as e:
                print "ENCODING ERROR:", line, e
                line = "{}"
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue

            if not choose(loaded_example):
                continue

            example = {}
            example["label"] = loaded_example["gold_label"]
            example["premise"] = loaded_example["sentence1"]
            example["hypothesis"] = loaded_example["sentence2"]
            example["example_id"] = loaded_example.get('pairID', 'NoID')
            example["get_length_average"] =get_length_average(example["premise"], example["hypothesis"])
            if loaded_example["sentence1_binary_parse"] and loaded_example["sentence2_binary_parse"]:
                (example["premise_tokens"], example["premise_transitions"]) = convert_binary_bracketing(
                    loaded_example["sentence1_binary_parse"], lowercase=lowercase)
                (example["hypothesis_tokens"], example["hypothesis_transitions"]) = convert_binary_bracketing(
                    loaded_example["sentence2_binary_parse"], lowercase=lowercase)
                examples.append(example)
            else:
                failed_parse += 1
    if failed_parse > 0:
        print(
            "Warning: Failed to convert binary parse for {} examples.".format(failed_parse))
    if level=="all" or level==2:
        return examples
    ne=np.array([e["get_length_average"] for e in examples])
    val=np.percentile(ne, 60)
    #return examples, val
    examples=np.array(examples)
    if level==1:
        examples=examples[(ne<val)]
    return list(examples)


if __name__ == "__main__":
    # Demo:
    examples = load_data('snli-data/snli_1.0_dev.jsonl')
    print examples[0]
