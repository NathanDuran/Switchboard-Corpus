class Dialogue:
    def __init__(self, file_name, conversation_num, num_utterances, utterances):
        self.file_name = file_name
        self.conversation_num = conversation_num
        self.num_utterances = num_utterances
        self.utterances = utterances

    def to_string(self):
        return str("File Name: " + self.file_name + "\n"
                   + "Conversation: " + str(self.conversation_num) + "\n"
                   + "Number of Utterances: " + str(self.num_utterances))


class Utterance:
    def __init__(self, speaker, text, da_label):
        self.speaker = speaker
        self.text = text
        self.da_label = da_label

    def to_string(self):
        return str(self.speaker + " " + self.text + " " + self.da_label)
