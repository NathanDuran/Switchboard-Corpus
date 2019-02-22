# Processing the Switchboard Dialogue Act Corpus

Utilities for processing the [Switchboard Dialogue Act Corpus](https://web.stanford.edu/~jurafsky/ws97/)
for the purpose of dialogue act classification. The data is split into the original [training](https://web.stanford.edu/~jurafsky/ws97/ws97-train-convs.list) 
and [test](https://web.stanford.edu/~jurafsky/ws97/ws97-test-convs.list) sets suggested by the authors.
The remaining dialogues have been used as an evaluation set and there is a further 300 dialogues from the training set for development purposes.

Thanks to Christopher Potts for providing the raw data in .csv format and the swda.py script for processing the .csv data, both of which can be found [here](https://github.com/cgpotts/swda)

## Data Format
Utterance are tagged with the [SWBD-DAMSL](https://web.stanford.edu/~jurafsky/ws97/manual.august1.html) dialogue acts.

The swda_to_text.py script processes all dialogues into a plain text format. Individual dialouges are saved into directories corresponding
to the set they belong to (train, test, etc). All utterances in a particular set are also saved to a text file.

The swda_utilities.py script contains various helper functions for loading/saving and processing the data, including a function for processing each dialogue.

By default:
- Utterances are written one per line in the format *Speaker* | *Utterance Text* | *Dialogue Act Tag*. This can be changed to only output the utterance text by setting the utterance_only_flag = True.
- Utterances marked as *Non-verbal* ('x' tags) are removed i.e. 'Laughter' or 'Throat_clearing'.
- Utterances marked as *Interrupted* ('+' tags) and continued later are concatenated to make un-interrupted sentences.
- All disfluency annotations are removed i.e. '#', '<', '>', etc.

### Example Utterances
A|What is the nature of your company's business?|qw

B|Well, it's actually, uh,|^h

B|we do oil well services.|sd

### Dialogue Acts
Dialogue Act    | Swda Label    | Count
--- | --- | ---
Statement-non-opinion   | sd    | 75136
Acknowledge (Backchannel)   | b | 38284
Statement-opinion   | sv    | 26421
Uninterpretable | %     | 15215
Agree/Accept    | aa    | 11123
Appreciation    | ba    | 4757
Yes-No-Question | qy    | 4725
Yes answers | ny    | 3031
Conventional-closing    | fc    | 2581
Wh-Question | qw | 1977
No answers  | nn | 1374
Response Acknowledgement    |bk | 1306
Hedge   | h  | 1226
Declarative Yes-No-Question | qy^d   | 1218
Backchannel in question form    | bh | 1053
Quotation   | ^q | 983
Summarize/reformulate   | bf | 952
Other   | fo_o_fw_"_by_bc    | 879
Affirmative non-yes answers | na    | 847
Action-directive    | ad    | 745
Collaborative Completion    | ^2    | 723
Repeat-phrase   | b^m   | 687
Open-Question   | qo    | 656
Open-Question   | qh    | 575
Hold before answer/agreement    | ^h    | 556
Reject  | ar    | 344
Negative non-no answers | ng    | 302
Signal-non-understanding    | br    | 298
Other answers   | no    | 285
Conventional-opening    | fp    | 225
Or-Clause   | qrr   | 209
Dispreferred answers    | arp_nd    | 207
3rd-party-talk  | t3    | 117
Offers, Options Commits | oo_co_cc  | 110
Maybe/Accept-part   | aap_am    | 104
Downplayer  | bd    | 103
Self-talk	| t1    | 103
Tag-Question    | ^g    | 92
Declarative Wh-Question | qw^d  | 80
Apology | fa    | 79
Thanking    | ft  | 78

## Metadata
The swda_metadata.py generates various metadata from the processed dialogues and saves them as a dictionary to a pickle file.
The words, labels and frequencies are also saved as plain text files in the /metadata directory.

- Total number of utterances:  199766
- Max utterance length:  133
- Maximum dialogue length: 457
- Vocabulary size: 22303
- Number of labels: 41
- Number of dialogue in train set: 1115
- Maximum length of dialogue in train set: 457
- Number of dialogue in test set: 19
- Maximum length of dialogue in test set: 330
- Number of dialogue in eval set: 21
- Maximum length of dialogue in eval set: 299
- Number of dialogue in dev set: 300
- Maximum length of dialogue in dev set: 405

### Keys and values for the metadata dictionary

- num_utterances = Total number of utterance in the full corpus.
- max_utterance_len = Number of words in the longest utterance in the corpus.
- max_dialogues_len = Number of utterances in the longest dialogue in the corpus
- word_freq = Dictionary with keys = words and values = frequencies
- vocabulary = Full vocabulary - Gluon NLP [Vocabulary](http://gluon-nlp.mxnet.io/api/modules/vocab.html#gluonnlp.Vocab)
- vocabulary_size = Number of words in the vocabulary.
- label_freq = Dictionary with keys = dialogue act labels and values = frequencies
- labels = Full labels - Gluon NLP [Vocabulary](http://gluon-nlp.mxnet.io/api/modules/vocab.html#gluonnlp.Vocab)
- num_labels = Number of labels used from the Switchboard data.

Each data set also has;
- *setname*_num_dialogues = Number of dialogues in the set
- *setname*_max_dialogues_len = Length of the longest dialogue in the set