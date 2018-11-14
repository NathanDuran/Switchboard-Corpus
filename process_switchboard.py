from swda import CorpusReader, Transcript
from utilities import *

# resource_dir = 'data/'
corpus = CorpusReader('switchboard_data/')

# Excluded dialogue act tags i.e. x = Non-verbal
excluded_tags = ['x']
# Excluded characters for ignoring i.e. <laughter>
excluded_chars = {'<', '>', '(', ')', '-'}  # ADD '#'???


transcript = Transcript('switchboard_data/sw00utt/sw_0001_4325.utt.csv', corpus.metadata)
print(transcript.swda_filename)


transcript_text = process_transcript(transcript, excluded_tags, excluded_chars)
# print(transcript_text)

# transcript_text = []
# for transcript in corpus.iter_transcripts(display_progress=False):
#     transcript_text = transcript_text + process_transcript(transcript, excluded_tags, excluded_chars)
#
# set_text = set(transcript_text)
# print(set_text)

