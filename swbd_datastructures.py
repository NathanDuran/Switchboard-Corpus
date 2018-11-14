class Utterance:
    def __init__(self, speaker, text, da_label):
        self.speaker = speaker
        self.text = text
        self.da_label = da_label

    def to_string(self):
        return str(self.speaker + " " + self.text + " " + self.da_label)