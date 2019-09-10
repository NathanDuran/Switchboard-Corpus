import itertools
import gluonnlp as nlp
from swda_utilities import *
# Initialise Spacy tokeniser
tokeniser = nlp.data.SpacyTokenizer('en')

# Dictionary for metadata
metadata = dict()

# Processed data directory
data_dir = 'swda_data'

# Metadata directory
metadata_dir = os.path.join(data_dir, 'metadata')

# Load full_set text file
swda_text = load_text_data(os.path.join(data_dir, 'full_set.txt'))

# Split into labels and utterances
utterances = []
for line in swda_text:
    utterances.append(line.split("|")[1])

# Count total number of utterances
num_utterances = len(utterances)

metadata['num_utterances'] = num_utterances
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

metadata['max_utterance_len'] = max_utterance_len
print("Max utterance length: ", max_utterance_len)
assert num_utterances == len(tokenised_utterances)

# Count the word frequencies and generate vocabulary
word_freq = nlp.data.count_tokens(list(itertools.chain(*tokenised_utterances)))
vocabulary = nlp.Vocab(word_freq)
vocabulary_size = len(word_freq)

metadata['word_freq'] = word_freq
metadata['vocabulary'] = vocabulary
metadata['vocabulary_size'] = vocabulary_size
print("Words:")
print(word_freq)
print(vocabulary)
print(vocabulary_size)

# Write vocabulary and word frequencies to file
with open(os.path.join(metadata_dir, 'vocabulary.txt'), 'w+') as file:
    for i in range(4, len(vocabulary)):
        file.write(vocabulary.to_tokens(i) + " " + str(word_freq[vocabulary.to_tokens(i)]) + "\n")

# Count the label frequencies and generate labels
labels, label_freq = get_label_frequency_distributions(data_dir, metadata_dir, label_index=2)
metadata['label_freq'] = label_freq
metadata['labels'] = labels
metadata['num_labels'] = len(labels)
print("Labels:")
print(labels)
print(len(labels))

# Create label frequency chart
label_freq_chart = plot_label_distributions(label_freq, title='Swda Label Frequency Distributions', num_labels=15)
label_freq_chart.savefig(os.path.join(metadata_dir, 'Swda Label Frequency Distributions.png'))

# Write labels and frequencies to file
save_label_frequency_distributions(label_freq, metadata_dir, 'labels.txt', to_markdown=False)

# Count sets number of dialogues and maximum dialogue length
max_dialogues_len = 0
sets = ['train', 'test', 'val', 'dev']
for i in range(len(sets)):

    # Load data set list
    set_list = load_text_data(os.path.join(metadata_dir, sets[i] + '_split.txt'))

    # Count the number of dialogues in the set
    set_num_dialogues = len(set_list)
    metadata[sets[i] + '_num_dialogues'] = set_num_dialogues
    print("Number of dialogue in " + sets[i] + " set: " + str(set_num_dialogues))

    # Count max number of utterances in sets dialogues
    set_max_dialogues_len = 0
    for dialogue in set_list:

        # Load dialogues utterances
        utterances = load_text_data(os.path.join(data_dir, sets[i], dialogue + '.txt'), verbose=False)

        # Check set and global maximum dialogue length
        if len(utterances) > set_max_dialogues_len:
            set_max_dialogues_len = len(utterances)

        if set_max_dialogues_len > max_dialogues_len:
            max_dialogues_len = set_max_dialogues_len

    metadata[sets[i] + '_max_dialogues_len'] = set_max_dialogues_len
    print("Maximum length of dialogue in " + sets[i] + " set: " + str(set_max_dialogues_len))

metadata['max_dialogues_len'] = max_dialogues_len
print("Maximum dialogue length: " + str(max_dialogues_len))

# Save data to pickle file
save_data_pickle(os.path.join(metadata_dir, 'metadata.pkl'), metadata)
