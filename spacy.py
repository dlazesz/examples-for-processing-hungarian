#!/usr/bin/env python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-

# Install this.
# !pip install https://github.com/oroszgy/spacy-hungarian-models/releases/download/hu_core_ud_lg-0.1.0/hu_core_ud_lg-0.1.0-py3-none-any.whl

import hu_core_ud_lg

def format_as_conllu(doc):
    """Orignal code: https://github.com/oroszgy/spacy-hungarian-models/blob/master/src/model_builder/io.py#L87"""
    res = []
    for sent in list(doc.sents):
        for i, word in enumerate(sent):
            if word.dep_.lower().strip() == "root":
                head_idx = 0
            else:
                head_idx = word.head.i - sent[0].i + 1

            linetuple = (
                str(i + 1),  # ID
                word.text,  # FORM
                word.lemma_,  # LEMMA
                word.pos_,  # UPOSTAG
                "_",  # XPOSTAG
                "_",  # FEATS
                str(head_idx),  # HEAD
                word.dep_,  # DEPREL
                "_",  # DEPS
                "_",  # MISC
            )
            res.append("\t".join(linetuple))

        res.append("")
    return "\n".join(res) + "\n"


nlp = hu_core_ud_lg.load()  # Load models

print(nlp.pipeline)  # Print loaded modules

nlp.remove_pipe("tagger")
nlp.remove_pipe("lemmatizer")
#nlp.remove_pipe("parser")    # Remove modules on demand
print(nlp.pipeline)

doc = nlp.make_doc('Ennek kell lennie a példamondatnak. Ez egy másik.')  # Create Document from text.
for name, proc in nlp.pipeline:             # iterate over components in order
    doc = proc(doc)
    print(name)

print(format_as_conllu(doc))  # Retrieve the result in CoNLL-U

"""
Remark:
As I see, currently the preprocessed input can be only handled when we "simulate the tagging and add the tags manually".
It is very complicated, so I obvously will not hack it together. 
SpaCy 2.1 is out it can export to JSON and soon there will be a newer version which can handle training material in JSON.
I expect that soon SpaCy will be able to handle preprocessed input combining these features.
The hungarian model is trained for 2.0 and either way one must train the modell again for 2.1 and newer. I leave this task for others.
"""
