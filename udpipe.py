#!/usr/bin/env python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-

# Install this.
# !pip install ufal.udpipe

# Load this
from ufal.udpipe import Model, Pipeline, ProcessingError

# Download model: https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2898
# Load this
model = Model.load('hungarian-szeged-ud-2.3-181115.udpipe')


pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')  # Full
pipeline2 = Pipeline(model, 'vertical', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu') # No tokenization, Vertical format as input...
pipeline3 = Pipeline(model, 'vertical', Pipeline.DEFAULT, Pipeline.NONE, 'conllu')    # No tokenization & No Parsing = Just POS tagging
pipeline4 = Pipeline(model, 'conllu', Pipeline.NONE, Pipeline.DEFAULT, 'conllu')      # Just Parsing...
# Remark: pipeline4 runs even when no POS taging supplied... Maybe it do POS tagging in background...

error = ProcessingError()  # For catching errors...

# Do the processing...
processed = pipeline.process('Az alma szép piros volt.', error)

assert error.occurred(), 'Error happened check documentation!'

# Write the output in CoNLL-U
print(processed)
