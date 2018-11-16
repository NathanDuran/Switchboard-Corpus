import itertools
import gluonnlp as nlp
from swda_utilities import *
# Initialise Spacy tokeniser
tokeniser = nlp.data.SpacyTokenizer('en')

# Processed data directory
data_dir = 'swda_data/'

# File with all swda data
all_swda_text_file = "all_swda_text.txt"

# Load all_swda text file
swda_text = load_data(data_dir + all_swda_text_file)

# Split into labels and utterances
utterances = []
labels = []
for line in swda_text:
    utterances.append(line.split("|")[1])
    labels.append(line.split("|")[2])

# utterances = ['Apple is looking at buying U.K. startup for $1 billion but spacey sucks.',
#             'So I am going to test wether they have stopped it slitting on but or whether it still suchs']

# Count total number of utterances
num_utterances = len(utterances)
print("Total number of utterances: ", num_utterances)

# Calculate max utterance length in tokens
max_utterance_len = 0
tokenised_utterances = []
for utt in utterances:

    # Tokenise utterance
    tokenised_utterance = tokeniser(utt)
    if len(tokenised_utterance) > max_utterance_len:
        max_utterance_len = len(tokenised_utterance)

    tokenised_utterances.append(tokenised_utterance)

print("Max utterance length: ", max_utterance_len)
assert num_utterances == len(tokenised_utterances)


# Count the word frequencies and generate vocabulary
word_freq = nlp.data.count_tokens(list(itertools.chain(*tokenised_utterances)))
vocabulary = nlp.Vocab(word_freq)
vocabulary_size = len(word_freq)
print("Words:")
print(word_freq)
print(vocabulary)
print(vocabulary_size)

# Count the label frequencies and generate labels
label_freq = nlp.data.count_tokens(labels)
labels = nlp.Vocab(label_freq)
num_labels = len(label_freq)
print("Labels:")
print(label_freq)
print(labels)
print(num_labels)

# Save data to file
data = dict(
    num_utterances=num_utterances,
    max_utterance_len=max_utterance_len,
    word_freq=word_freq,
    vocabulary=vocabulary,
    vocabulary_size=vocabulary_size,
    label_freq=labels,
    labels=labels,
    num_labels=num_labels)

save_data(data_dir + "metadata.pkl", data)