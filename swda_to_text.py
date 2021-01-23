import zipfile
import tempfile
from swda import CorpusReader
from utilities import *
from process_transcript import *

# Switchboard archive directory
archive_dir = 'swda_archive'

# Processed data directory
data_dir = 'swda_data'

# Metadata directory
metadata_dir = os.path.join(data_dir, 'metadata')

# If flag is set will only write utterances and not speaker or DA label
utterance_only_flag = False

# Excluded dialogue act tags i.e. x = Non-verbal
excluded_tags = ['x']
# Excluded characters for ignoring i.e. <laughter>
excluded_chars = {'<', '>', '(', ')', '-', '#'}

# Load training, test, validation and development splits
train_split = load_text_data(os.path.join(metadata_dir, 'train_split.txt'))
test_split = load_text_data(os.path.join(metadata_dir, 'test_split.txt'))
val_split = load_text_data(os.path.join(metadata_dir, 'val_split.txt'))

# Files for all the utterances in the corpus and data splits
full_set_file = "full_set"
train_set_file = "train_set"
test_set_file = "test_set"
val_set_file = "val_set"

# Remove old files if they exist, so we do not append to old data
remove_file(data_dir, full_set_file, utterance_only_flag)
remove_file(data_dir, train_set_file, utterance_only_flag)
remove_file(data_dir, test_set_file, utterance_only_flag)
remove_file(data_dir, val_set_file, utterance_only_flag)

# Create a temporary directory and unzip the archived data
with tempfile.TemporaryDirectory(dir=archive_dir) as tmp_dir:
    print('Created temporary directory', tmp_dir)

    zip_file = zipfile.ZipFile(os.path.join(archive_dir, 'swda_archive.zip'), 'r')
    zip_file.extractall(tmp_dir)
    zip_file.close()

    # Corpus object for iterating over the whole corpus in .csv format
    corpus = CorpusReader(tmp_dir)

    # Process each transcript
    for transcript in corpus.iter_transcripts(display_progress=False):

        # Process the utterances and create a dialogue object
        dialogue = process_transcript(transcript, excluded_tags, excluded_chars)

        # Append all utterances to full_set_file text file
        dialogue_to_file(os.path.join(data_dir, full_set_file), dialogue, utterance_only_flag, 'a+')

        # Determine which set this dialogue belongs to (training, test or validation)
        set_dir = ''
        set_file = ''
        if dialogue.conversation_id in train_split:
            set_dir = 'train'
            set_file = train_set_file
        elif dialogue.conversation_id in test_split:
            set_dir = 'test'
            set_file = test_set_file
        elif dialogue.conversation_id in val_split:
            set_dir = 'val'
            set_file = val_set_file

        # If only saving utterances use different directory
        if utterance_only_flag:
            set_dir = os.path.join(data_dir, set_dir + '_utt')
        else:
            set_dir = os.path.join(data_dir, set_dir)

        # Create the directory if is doesn't exist yet
        if not os.path.exists(set_dir):
            os.makedirs(set_dir)

        # Write individual dialogue to train, test or validation folders
        dialogue_to_file(os.path.join(set_dir, dialogue.conversation_id), dialogue, utterance_only_flag, 'w+')

        # Append all dialogue utterances to sets file
        dialogue_to_file(os.path.join(data_dir, set_file), dialogue, utterance_only_flag, 'a+')

