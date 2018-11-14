from swbd_datastructures import *

def process_transcript(transcript, excluded_tags=None, excluded_chars=None):

    utterances = []

    for utt in transcript.utterances:

        utterance_text = []

        for word in utt.text_words(filter_disfluency=True):

            # Remove the annotations that filter_disfluency does not (i.e. <laughter>)
            # If no excluded characters are present just add it
            if all(char not in excluded_chars for char in word):
                utterance_text.append(word)
            # Else, for hyphenated words, check first, last and 2nd to last char for interrupted words (i.e. 'spi-,')
            elif len(word) > 1:
                if word[0] not in excluded_chars and word[-1] not in excluded_chars and word[-2] not in excluded_chars:
                    utterance_text.append(word)

        # Join words for complete sentence
        utterance_text = " ".join(utterance_text)

        # Print original and processed utterances
        # print(utt.transcript_index, " ", utt.text_words(filter_disfluency=True), " ", utt.damsl_act_tag())
        # print(utt.transcript_index, " ", utterance_text, " ", utt.damsl_act_tag())

        # Check we are not adding an empty utterance (i.e. because it was just <laughter>),
        # or adding an utterance with an excluded tag.
        if len(utterance_text) > 0 and utt.damsl_act_tag() not in excluded_tags:
            # Add utterance to list
            current_utt = Utterance(utt.caller, utterance_text, utt.damsl_act_tag())
            utterances.append(current_utt)

            print(current_utt.to_string())
        else:
            print("removing ", utt.utterance_index, " ", utt.text_words(filter_disfluency=True), " with tag", utt.damsl_act_tag())

    print(len(utterances))

    # Concatenate multi-utterance's with '+' label
    current_a = None
    current_b = None
    for utt in reversed(utterances):

        # If we find an utterance that must be concatenated
        if utt.da_label == '+':
            # Save to temp variable
            if utt.speaker == 'A':
                current_a = utt.text
            else:
                current_b = utt.text

            # And remove from list
            utterances.remove(utt)

        # Else if we have an utterance to concatenate
        elif current_a and utt.speaker == 'A':
            print("concat '", utt.text, "' with '", current_a, "'")
            utt.text = utt.text + current_a
            current_a = None

        elif current_b and utt.speaker == 'B':
            print("concat '", utt.text, "' with '", current_b, "'")
            utt.text = utt.text + current_b
            current_b = None

    print(len(utterances))

    for i in range(len(utterances)):
        print((i + 1), utterances[i].to_string())

    # return transcript_data
