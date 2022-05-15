![](https://github.com/themichaelmort/machine_tramslation/blob/main/Machine-Translation.gif)

# Machine Translation

Twice a year, the text of all speeches made at the General Conference for The Church of Jesus-Christ of Latter-day Saints are translated from English into Spanish. Using a corpus of translated text, a transformer was trained with multi-headed attention to translate from Spanish into English. In general, the model is capable to rendering human-understandable translations. The model performs well enough that discrepencies in the translated text and the original English are often due to translation differences introduced by human translators moving from English to Spanish rather than the machine translation from Spanich back into English. See the graphic above for one such example.

## At a Glance

* Dataset - General Conference 2017 text in ![English](https://raw.githubusercontent.com/nickwalton/translation/master/gc_2010-2017_conglomerated_20171009_en.txt) and ![Spanish](https://raw.githubusercontent.com/nickwalton/translation/master/gc_2010-2017_conglomerated_20171009_es.txt). As part of the transformer architecture, dataset is encoded with positional encoding so that it can be used with the attention mechanism.
* Model - Transformer architecture, as outlined in ["Attention is All you Need"](https://arxiv.org/pdf/1706.03762.pdf) and ["The Annotated Transformer"](http://nlp.seas.harvard.edu/annotated-transformer/). The transformer consists of an encoder and decoder unit. The benefit of transformers over other sequential models like LSTMs or RNNs is that transformers do not suffer from the short term memory issues of other sequential models.
* The Python script in this project was originally drafted in Google Colab. Much of it was adapted from the resources mentioned below for the purpose of learning about transformer models.

## Resources
I am indebted to Harvard NLP's ["The Annotated Transformer"](http://nlp.seas.harvard.edu/annotated-transformer/), a publically available PyTorch transformer implementation that served as a starting point for this project. Also, I relied upon section 3.2 of the paper ["Attention is All you Need"](https://arxiv.org/pdf/1706.03762.pdf) by Vaswani et al., to learn about attention.
