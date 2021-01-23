import itertools
from utilities import *
# Initialise Spacy tokeniser
tokeniser = nlp.data.SpacyTokenizer('en_core_web_sm')

# Dictionary for metadata
metadata = dict()

# Processed data directory
data_dir = 'swda_data'

# Metadata directory
metadata_dir = os.path.join(data_dir, 'metadata')

# Load full_set text file
text = load_text_data(os.path.join(data_dir, 'full_set.txt'))

# Split into speakers and utterances
utterances = []
speakers = []
for line in text:
    utterances.append(line.split("|")[1])
    speakers.append(line.split("|")[0])

# Count total number of utterances
num_utterances = len(utterances)
metadata['num_utterances'] = num_utterances

# Calculate max/mean utterance length in tokens
max_utterance_len = 0
mean_utterance_len = 0
tokenised_utterances = []
for utt in utterances:

    # Tokenise utterance
    tokenised_utterance = tokeniser(utt)

    # Remove whitespace tokens
    tokenised_utterance = [token if not token.isspace() else '' for token in tokenised_utterance]

    if len(tokenised_utterance) > max_utterance_len:
        max_utterance_len = len(tokenised_utterance)

    tokenised_utterances.append(tokenised_utterance)

    # Add length to mean
    mean_utterance_len += len(tokenised_utterance)

assert num_utterances == len(tokenised_utterances)
metadata['max_utterance_len'] = max_utterance_len
metadata['mean_utterance_len'] = mean_utterance_len / num_utterances

# Count each sets number of dialogues, max/mean dialogue length and number of utterances
max_dialogue_len = 0
mean_dialogue_len = 0
num_dialogues = 0
sets = ['train', 'test', 'val']
for dataset_name in sets:

    # Load data set list
    set_list = load_text_data(os.path.join(metadata_dir, dataset_name + '_split.txt'))

    # Count the number of dialogues in the set
    set_num_dialogues = len(set_list)
    metadata[dataset_name + '_num_dialogues'] = set_num_dialogues

    # Count max number of utterances in sets dialogues
    set_max_dialogue_len = 0
    set_mean_dialogue_len = 0
    set_num_utterances = 0
    for dialogue in set_list:

        # Load dialogues utterances
        utterances = load_text_data(os.path.join(data_dir, dataset_name, dialogue + '.txt'), verbose=False)

        # Count dialogue length for means/number of utterances
        num_dialogues += 1
        mean_dialogue_len += len(utterances)
        set_mean_dialogue_len += len(utterances)
        set_num_utterances += len(utterances)

        # Check set and global maximum dialogue length
        if len(utterances) > set_max_dialogue_len:
            set_max_dialogue_len = len(utterances)

        if set_max_dialogue_len > max_dialogue_len:
            max_dialogue_len = set_max_dialogue_len

    metadata[dataset_name + '_max_dialogue_len'] = set_max_dialogue_len
    metadata[dataset_name + '_mean_dialogue_len'] = set_mean_dialogue_len / set_num_dialogues
    metadata[dataset_name + '_num_utterances'] = set_num_utterances

metadata['num_dialogues'] = num_dialogues
metadata['max_dialogue_len'] = max_dialogue_len
metadata['mean_dialogue_len'] = mean_dialogue_len / num_dialogues

# Count the word frequencies and generate vocabulary
word_freq = pd.DataFrame.from_dict(nlp.data.count_tokens(list(itertools.chain(*tokenised_utterances))), orient='index')
word_freq.reset_index(inplace=True)
word_freq.columns = ['Words', 'Count']
word_freq.sort_values('Count', ascending=False, ignore_index=True, inplace=True)
vocabulary = word_freq['Words'].to_list()
vocabulary_size = len(word_freq)

metadata['word_freq'] = word_freq
metadata['vocabulary'] = vocabulary
metadata['vocabulary_size'] = vocabulary_size
print("Vocabulary:")
print(word_freq)
print(vocabulary)

# Write vocabulary and word frequencies to file
save_word_frequency_distributions(word_freq, metadata_dir, 'word_freq.txt')
save_text_data(os.path.join(metadata_dir, 'vocabulary.txt'), vocabulary)

# Count the label frequencies and generate labels
labels, label_freq = get_label_frequency_distributions(data_dir, metadata_dir, label_index=2)
metadata['label_freq'] = label_freq
metadata['labels'] = labels
metadata['num_labels'] = len(labels)
print("Labels:")
print(labels)

# Create label frequency chart
label_freq_chart = plot_label_distributions(label_freq, title='Swda Label Frequency Distributions', num_labels=15)
label_freq_chart.savefig(os.path.join(metadata_dir, 'Swda Label Frequency Distributions.png'))

# Write labels and frequencies to file
save_label_frequency_distributions(label_freq, metadata_dir, 'label_freq.txt', to_markdown=False)
save_text_data(os.path.join(metadata_dir, 'labels.txt'), labels)

# Count speakers and save to list
metadata['num_speakers'] = len(set(speakers))
save_text_data(os.path.join(metadata_dir, 'speakers.txt'), list(set(speakers)))

# Create and print the metadata string
metadata_str = ["- Total number of utterances: " + str(metadata['num_utterances']),
                "- Max utterance length: " + str(metadata['max_utterance_len']),
                "- Mean utterance length: " + str(round(metadata['mean_utterance_len'], 2)),
                "- Total Number of dialogues: " + str(metadata['num_dialogues']),
                "- Max dialogue length: " + str(metadata['max_dialogue_len']),
                "- Mean dialogue length: " + str(round(metadata['mean_dialogue_len'], 2)),
                "- Vocabulary size: " + str(metadata['vocabulary_size']),
                "- Number of labels: " + str(metadata['num_labels']),
                "- Number of speakers: " + str(metadata['num_speakers'])]

for dataset_name in sets:
    metadata_str.append(dataset_name.capitalize() + " set")
    metadata_str.append("- Number of dialogues: " + str(metadata[dataset_name + '_num_dialogues']))
    metadata_str.append("- Max dialogue length: " + str(metadata[dataset_name + '_max_dialogue_len']))
    metadata_str.append("- Mean dialogue length: " + str(round(metadata[dataset_name + '_mean_dialogue_len'], 2)))
    metadata_str.append("- Number of utterances: " + str(metadata[dataset_name + '_num_utterances']))

for string in metadata_str:
    print(string)

# Save metadata to pickle and text file
save_data_pickle(os.path.join(metadata_dir, 'metadata.pkl'), metadata)
save_text_data(os.path.join(metadata_dir, 'metadata.txt'), metadata_str)