import os
import pickle
import gluonnlp as nlp
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def load_text_data(path, verbose=False):
    with open(path, "r") as file:
        # Read a line and strip newline char
        lines = [line.rstrip('\r\n') for line in file.readlines()]
    if verbose:
        print("Loaded data from file %s." % path)
    return lines


def save_data_pickle(path, data, verbose=True):
    with open(path, "wb") as file:
        pickle.dump(data, file, protocol=2)
    if verbose:
        print("Saved data to file %s." % path)


def dialogue_to_file(path, dialogue, utterance_only, write_type):
    if utterance_only:
        path = path + "_utt"
    with open(path + ".txt", write_type) as file:
        for utterance in dialogue.utterances:
            if utterance_only:
                file.write(utterance.text.strip() + "\n")
            else:
                file.write(utterance.speaker + "|" + utterance.text.strip() + "|" + utterance.da_label + "\n")


def remove_file(data_dir, file, utterance_only):
    # Remove either text or full versions
    if utterance_only:
        if os.path.exists(os.path.join(data_dir, file + '_utt.txt')):
            os.remove(os.path.join(data_dir, file + '_utt.txt'))
    else:
        if os.path.exists(os.path.join(data_dir, file + '.txt')):
            os.remove(os.path.join(data_dir, file + '.txt'))


def get_label_frequency_distributions(data_dir, metadata_dir, label_map='label_map.txt', label_index=2):
    # Load set text files and get the labels
    full_set = load_text_data(os.path.join(data_dir, 'full_set.txt'))
    train_set = load_text_data(os.path.join(data_dir, 'train_set.txt'))
    test_set = load_text_data(os.path.join(data_dir, 'test_set.txt'))
    val_set = load_text_data(os.path.join(data_dir, 'val_set.txt'))
    full_labels = [line.split("|")[label_index] for line in full_set]
    train_labels = [line.split("|")[label_index] for line in train_set]
    test_labels = [line.split("|")[label_index] for line in test_set]
    val_labels = [line.split("|")[label_index] for line in val_set]

    # Get the label frequencies
    full_freq = nlp.data.count_tokens(full_labels)
    train_freq = nlp.data.count_tokens(train_labels)
    test_freq = nlp.data.count_tokens(test_labels)
    val_freq = nlp.data.count_tokens(val_labels)

    # Get the full label list
    labels = sorted(full_freq, key=full_freq.get, reverse=True)

    # Get the label names from the labels map
    label_names = load_text_data(os.path.join(metadata_dir, label_map))
    label_names = [line.split("|")[0] for line in label_names]

    # Get the count and percentage values
    full_count = [full_freq[label] for label in labels]
    full_perc = [(100 / len(full_labels)) * full_freq[label] for label in labels]

    train_count = [train_freq[label] for label in labels]
    train_perc = [(100 / len(train_labels)) * train_freq[label] for label in labels]

    test_count = [test_freq[label] for label in labels]
    test_perc = [(100 / len(test_labels)) * test_freq[label] for label in labels]

    val_count = [val_freq[label] for label in labels]
    val_perc = [(100 / len(val_labels)) * val_freq[label] for label in labels]

    # Create dataframe
    label_freq = pd.DataFrame({'Dialogue Act': label_names, 'Labels': labels,
                               'Count': full_count, '%': full_perc,
                               'Train Count': train_count, 'Train %': train_perc,
                               'Test Count': test_count, 'Test %': test_perc,
                               'Val Count': val_count, 'Val %': val_perc})

    return labels, label_freq


def save_label_frequency_distributions(data, metadata_dir, file_name, to_markdown=False):
    if not to_markdown:
        header_format = '{:30} {:^20} {:^8} {:^8} {:^15} {:^8} {:^15} {:^8} {:^15} {:^8}\n'
        row_format = '{:30} {:^20} {:^8d} {:^8.2f} {:^15d} {:^8.2f} {:^15d} {:^8.2f} {:^15d} {:^8.2f}\n'
    else:
        header_format = '{:30} | {:^20} | {:^8} | {:^8} | {:^15} | {:^8} | {:^15} | {:^8} | {:^15} | {:^8}\n'
        row_format = '{:30} | {:^20} | {:^8d} | {:^8.2f} | {:^15d} | {:^8.2f} | {:^15d} | {:^8.2f} | {:^15d} | {:^8.2f}\n'

    with open(os.path.join(metadata_dir, file_name), 'w+') as file:
        for index, row in data.iterrows():
            if index == 0:
                file.write(header_format.format(*data.columns.to_list()))
                if to_markdown:
                    file.write('--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---:\n')
            file.write(row_format.format(row['Dialogue Act'], row['Labels'],
                                         row['Count'], row['%'],
                                         row['Train Count'], row['Train %'],
                                         row['Test Count'], row['Test %'],
                                         row['Val Count'], row['Val %']))


def plot_label_distributions(data, num_labels=None, title=None, fig_size=(10, 10), font_size=15):
    # Reshape dataframe for plotting
    data_reshaped = data.drop(['Dialogue Act', 'Count', '%', 'Train Count', 'Test Count', 'Val Count'], axis=1)
    if num_labels:
        data_reshaped = data_reshaped.head(num_labels)
    data_reshaped = data_reshaped.melt(id_vars='Labels', var_name='set', value_name='percentage')

    # Create figure and barchart
    fig = plt.figure(figsize=fig_size)
    sns.barplot(x='Labels', y='percentage', hue='set', data=data_reshaped)
    sns.despine(left=True, bottom=True)

    # Set legend and axis labels
    plt.legend(ncol=1, loc="upper right", frameon=True)
    plt.ylabel('%', fontsize=font_size)
    plt.xlabel('Labels', fontsize=font_size)
    if title:
        plt.title(title, fontsize=font_size)

    return fig
