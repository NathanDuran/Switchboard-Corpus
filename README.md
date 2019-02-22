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

## Metadata

