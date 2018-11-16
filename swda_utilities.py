import os


class Dialogue:
    def __init__(self, folder_name, conversation_num, num_utterances, utterances):
        self.folder_name = folder_name
        self.conversation_num = conversation_num
        self.num_utterances = num_utterances
        self.utterances = utterances

    def to_string(self):
        return str("Folder Name: " + self.folder_name + "\n"
                   + "Conversation: " + self.conversation_num + "\n"
                   + "Number of Utterances: " + str(self.num_utterances))


class Utterance:
    def __init__(self, speaker, text, da_label):
        self.speaker = speaker
        self.text = text
        self.da_label = da_label

    def to_string(self):
        return str(self.speaker + " " + self.text + " " + self.da_label)


def process_transcript(transcript, excluded_tags=None, excluded_chars=None):

    # Process each utterance in the transcript and create list of Utterance objects
    utterances = []
    for utt in transcript.utterances:

        # Remove the word annotations that filter_disfluency does not (i.e. <laughter>)
        utterance_text = []
        for word in utt.text_words(filter_disfluency=True):

            # If no excluded characters are present just add it
            if all(char not in excluded_chars for char in word):
                utterance_text.append(word)
            # Else, if it contains'#' that is sometimes appended to words remove
            elif any(char is '#' for char in word):
                word = word.replace('#', "")
                utterance_text.append(word)
            # Else, to keep hyphenated words, check 1st, last and 2nd-to-last char for interruptions (i.e. 'spi-,')
            elif len(word) > 1:
                if word[0] not in excluded_chars and word[-1] not in excluded_chars and word[-2] not in excluded_chars:
                    utterance_text.append(word)

        # Join words for complete sentence
        utterance_text = " ".join(utterance_text)
        # Strip leading and trailing whitespace
        utterance_text.strip()

        # Print original and processed utterances
        # print(utt.transcript_index, " ", utt.text_words(filter_disfluency=True), " ", utt.damsl_act_tag())
        # print(utt.transcript_index, " ", utterance_text, " ", utt.damsl_act_tag())

        # Check we are not adding an empty utterance (i.e. because it was just <laughter>),
        # or adding an utterance with an excluded tag.
        if len(utterance_text) > 0 and utt.damsl_act_tag() not in excluded_tags:
            # Create Utterance and add to list
            current_utt = Utterance(utt.caller, utterance_text, utt.damsl_act_tag())
            utterances.append(current_utt)

    # Concatenate multi-utterance's with '+' label
    current_a = None
    current_b = None
    for utt in reversed(utterances):

        # If we find an utterance that must be concatenated
        if utt.da_label == '+':
            # Save to temp variable
            if utt.speaker == 'A':
                # Need to check if we have multiple lines to concatenate
                if current_a:
                    #current_a += utt.text
                    current_a = utt.text + " " + current_a
                else:
                    current_a = utt.text

            elif utt.speaker == 'B':
                if current_b:
                    current_b = utt.text + " " + current_b
                else:
                    current_b = utt.text

            # And remove utterance from list
            utterances.remove(utt)

        # Else if we have an utterance to concatenate
        elif current_a and utt.speaker == 'A':
            # Add it to the utterance and set temp empty
            utt.text = utt.text + " " + current_a
            current_a = None
            # print("Concatenating '", utt.text, "' + '", current_a, "'")
        elif current_b and utt.speaker == 'B':
            utt.text = utt.text + " " + current_b
            current_b = None
            # print("Concatenating '", utt.text, "' + '", current_b, "'")

    # Create Dialogue
    folder_name = transcript.swda_filename.split('\\')[1] # Trim to just folder name
    conversation_num = str(transcript.utterances[0].conversation_no)
    dialogue = Dialogue(folder_name, conversation_num, len(utterances), utterances)

    return dialogue


def read_file(path, verbose=True):
    with open(path, "r") as file:
        # Read a line and strip newline char
        lines = [line.rstrip('\r\n') for line in file.readlines()]
    if verbose:
        print("Loaded data from file %s." % path)
    return lines


def append_to_file(path, dialogue):
    with open(path, 'a+') as file:
        for utterance in dialogue.utterances:
            file.write(utterance.speaker + "|" + utterance.text + "|" + utterance.da_label + "\n")


def remove_file(data_dir, file):
    if os.path.exists(data_dir + file):
        os.remove(data_dir + file)

