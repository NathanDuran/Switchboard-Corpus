from swda import CorpusReader
from swda_utilities import *

# Processed data directory
data_dir = 'swda_data/'

# Corpus object for iterating over the whole corpus in .csv format
corpus = CorpusReader('raw_swda_data/')

# Load training, test, validation and development splits
train_split = load_data(data_dir + 'train_split.txt')
test_split = load_data(data_dir + 'test_split.txt')
val_split = load_data(data_dir + 'eval_split.txt')
dev_split = load_data(data_dir + 'dev_split.txt')

# Files for all the utterances in the corpus and data splits
all_swda_text_file = "all_swda_text.txt"
train_split_text_file = "train_text.txt"
test_split_text_file = "test_text.txt"
val_split_text_file = "eval_text.txt"
dev_split_text_file = "dev_text.txt"

# Remove old files if they exist, so we do not append to old data
remove_file(data_dir, all_swda_text_file)
remove_file(data_dir, train_split_text_file)
remove_file(data_dir, test_split_text_file)
remove_file(data_dir, val_split_text_file)
remove_file(data_dir, dev_split_text_file)

# Excluded dialogue act tags i.e. x = Non-verbal
excluded_tags = ['x']
# Excluded characters for ignoring i.e. <laughter>
excluded_chars = {'<', '>', '(', ')', '-', '#'}

# Process each transcript
for transcript in corpus.iter_transcripts(display_progress=False):

    # Process the utterances and create a dialogue object
    dialogue = process_transcript(transcript, excluded_tags, excluded_chars)

    # Write all utterances to all_swda_text_file
    append_to_file(data_dir + all_swda_text_file, dialogue)

    # Determine which set this dialogue belongs to (training, test or evaluation)
    set_dir = ''
    set_file = ''
    if dialogue.conversation_num in train_split:
        set_dir = data_dir + 'train/'
        set_file = train_split_text_file
    elif dialogue.conversation_num in test_split:
        set_dir = data_dir + 'test/'
        set_file = test_split_text_file
    elif dialogue.conversation_num in val_split:
        set_dir = data_dir + 'eval/'
        set_file = val_split_text_file

    # Create the directory if is doesn't exist yet
    if not os.path.exists(set_dir):
        os.makedirs(set_dir)

    # Write individual dialogue to train, test or validation folders
    with open(set_dir + dialogue.conversation_num + ".txt", 'w+') as file:
        for utterance in dialogue.utterances:
            file.write(utterance.speaker + "|" + utterance.text + "|" + utterance.da_label + "\n")

    # Write all dialogue utterances to sets file
    append_to_file(data_dir + set_file, dialogue)

    # If it is also in the development set write it there too
    if dialogue.conversation_num in dev_split:

        set_dir = data_dir + 'dev/'
        set_file = dev_split_text_file

        # Create the directory if is doesn't exist yet
        if not os.path.exists(set_dir):
            os.makedirs(set_dir)

        # Write individual dialogue to dev folder
        with open(set_dir + dialogue.conversation_num + ".txt", 'w+') as file:
            for utterance in dialogue.utterances:
                file.write(utterance.speaker + "|" + utterance.text + "|" + utterance.da_label + "\n")

        # Write all dialogue utterances to dev set file
        append_to_file(data_dir + set_file, dialogue)
