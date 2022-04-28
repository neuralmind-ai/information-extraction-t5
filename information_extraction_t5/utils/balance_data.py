"""Code to balance the dataset, keeping a negative-positive ratio"""
import numpy as np
import pandas as pd

from information_extraction_t5.features.sentences import split_t5_sentence_into_components


def count_pos_neg(labels, document_ids, example_ids):
    """Count the number of positive and negative samples. The counting is also
    done for each pair of document_id-example-id returned as a dict."""

    pos, neg = 0, 0
    counter = {}
    for label, document_id, example_id in zip(labels, document_ids, example_ids):
        if document_id not in counter:
            counter[document_id] = {}

        if example_id not in counter[document_id].keys():
            counter[document_id][example_id] = {'pos': 0, 'neg': 0}

        if 'N/A' in label:
            counter[document_id][example_id]['neg'] += 1
            neg += 1
        else:
            counter[document_id][example_id]['pos'] += 1
            pos += 1
    return pos, neg, counter


def balance_data(inputs, labels, document_ids, example_ids, negative_ratio):
    """Control the number of negative examples (N/A) with respect to the number
    of positive examples by negative_ratio. The data balancing is performed for
    each pair of document_id-example_id.

    Negative samples are selected with resampling with replacement, as the number
    of negative samples can be lower than the number of positives * negative_ratio.
    """
    
    n_pos, n_neg, _ = count_pos_neg(labels, document_ids, example_ids)
    print(f'>> The negative-positive ratio of the original dataset is {n_neg/n_pos:.2f}.')

    is_negative = []
    for label in labels:
        _, _, sub_answers = split_t5_sentence_into_components(label)
        if 'N/A' in sub_answers:
            is_negative.append(True)
        else:
            is_negative.append(False)

    # create the initial dataframe
    arr = np.vstack([
        np.array(inputs),
        np.array(labels),
        np.array(document_ids),
        np.array(example_ids),
        np.array(is_negative, dtype=bool)]).transpose()
    df1 = pd.DataFrame(arr, columns=['examples', 'labels', 'document_ids', 'example_ids', 'is_negative'])

    # Separate positive and negative samples
    df_pos = df1.loc[df1['is_negative'] == 'False']
    df_neg = df1.loc[df1['is_negative'] == 'True']

    # create temporary dataframe with additional column to count how many
    # positive qas we have for each pair document_id-example_id
    df_pos_counter = df_pos.groupby(['document_ids', 'example_ids']).size().reset_index(name='counts')

    # merge the positive dataframe with counter and negative dataframe
    df_merge = df_pos_counter.merge(df_neg, on=['document_ids', 'example_ids'], how='outer')
    # remove pairs document x example that has only negative (no positite qa)
    df_merge.dropna(inplace=True)

    # process the merged dataframe to resample negative cases proportional
    # to the number of positive cases
    df_group = df_merge.groupby(['document_ids', 'example_ids'])
    frames = []
    for group in df_group.groups:
        df = df_group.get_group(group)
        df = df.sample(int(df['counts'].values[0]) * negative_ratio, replace=True, random_state=42)
        frames.append(df)
    df_merge = pd.concat(frames)

    # remove temporary columns
    df_merge = df_merge.drop(['counts', 'is_negative'], axis=1)
    df_pos = df_pos.drop(['is_negative'], axis=1)

    # create the final dataframe by concatenating positives and negatives
    dfinal = pd.concat([df_pos, df_merge])

    inputs, labels, document_ids, example_ids = dfinal.T.values.tolist()

    n_pos, n_neg, _ = count_pos_neg(labels, document_ids, example_ids)
    if n_neg/n_pos != negative_ratio:
        print(f'>> The resultant negative-positive ratio is {n_neg/n_pos:.2f}. '
              'Hint: set "use_missing_answers=False" to get a precise data balancing.')
    else:
        print(f'>> The resultant negative-positive ratio is {n_neg/n_pos:.2f}.')

    return inputs, labels, document_ids, example_ids
