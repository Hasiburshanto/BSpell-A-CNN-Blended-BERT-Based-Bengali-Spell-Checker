# BSpell: A CNN-Blended BERT Based Bengali Spell Checker
Bengali typing is mostly performed using English keyboard and can be highly erroneous due to the presence of compound and
similarly pronounced letters. Spelling correction of a misspelled word requires understanding of word typing pattern as well as the
context of the word usage. A specialized BERT model named BSpell has been proposed in this paper targeted towards word for word
correction in sentence level. BSpell contains an end-to-end trainable CNN sub-model named SemanticNet along with specialized
auxiliary loss. This allows BSpell to specialize in highly inflected Bengali vocabulary in the presence of spelling errors. furthermore, a
hybrid pretraining scheme has been proposed for BSpell that combines word level and character level masking. Comparison on two
Bengali and one Hindi spelling correction dataset shows the superiority of our proposed approach.

Cite the code : [![DOI](https://zenodo.org/badge/606186668.svg)](https://zenodo.org/badge/latestdoi/606186668)

Dataset : [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7675570.svg)](https://doi.org/10.5281/zenodo.7675570)
