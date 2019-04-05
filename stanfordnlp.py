#!/usr/bin/env python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-

# Install this.
# !pip install stanfordnlp

# Load this
import stanfordnlp
from stanfordnlp.models.common import conll

# This must run only once...
stanfordnlp.download('hu_szeged')

nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos,lemma,depparse', lang="hu")  # Full pipeline
nlp1 = stanfordnlp.Pipeline(processors='tokenize,mwt', lang="hu")                    # Part I.
nlp2 = stanfordnlp.Pipeline(processors='pos,lemma,depparse', lang="hu")              # Part II.

# Analyze raw string
doc = nlp1('Kecském kucorog, macskám mocorog.')

# Print result...
for i in range(len(doc.sentences)):
  doc.sentences[i].print_tokens()

conllu_format = doc.conll_file.conll_as_string()
print(conllu_format)  # CoNLL text output...

# Documentation: https://stanfordnlp.github.io/stanfordnlp/processors.html

# Read CoNLL-U in any stage...
doc = stanfordnlp.Document(None)
doc.conll_file = conll.CoNLLFile(input_str=conllu_format)

# Analyze further and print the result...
doc2 = nlp2(doc)
print(doc2.conll_file.conll_as_string())
