import os
import numpy as np
import pandas as pd

from information_extraction_t5.features.sentences import split_t5_sentence_into_components


def count_pos_neg(inputs, labels, document_ids, example_ids):
	""""Count the number of positive and negative samples. The counting is also
	done for each pair of document_id-example-id returned as a dict."""

	pos, neg = 0, 0
	counter = {}
	for input, label, document_id, example_id in zip(inputs, labels, document_ids, example_ids):
		if document_id not in counter.keys():
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

	n_pos, n_neg, _ = count_pos_neg(inputs, labels, document_ids, example_ids)
	print('>> The negative-positive ratio of the original dataset is {:.2f}.'.format(n_neg/n_pos))

	is_negative = []
	for label in labels:
		_, _, sub_answers = split_t5_sentence_into_components(label)
		if 'N/A' in sub_answers:
			is_negative.append(True)
		else:
			is_negative.append(False)

	# create the initial dataframe
	arr = np.vstack([np.array(inputs), np.array(labels), np.array(document_ids),
					np.array(example_ids), np.array(is_negative, dtype=bool)]).transpose()
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

	n_pos, n_neg, _ = count_pos_neg(inputs, labels, document_ids, example_ids)
	if n_neg/n_pos != negative_ratio:
		print('>> The resultant negative-positive ratio is {:.2f}. '
			  'Hint: set "use_missing_answers=False" to get a precise data balancing.'
			  .format(n_neg/n_pos))
	else:
		print('>> The resultant negative-positive ratio is {:.2f}.'.format(n_neg/n_pos))

	return inputs, labels, document_ids, example_ids


def main():
	from transformers.data.processors.squad import SquadV1Processor
	from rich.progress import track

	from neural_extract.features.preprocess import generate_t5_input_sentence, generate_t5_label_sentence
	
	train_file = 'data/processed/test-v0.6.json'
	# train_file = 'data/processed/test-v0.6-10windows.json'
	negative_ratio = 2

	processor = SquadV1Processor()
	examples = processor.get_dev_examples(os.path.dirname(train_file), filename=os.path.basename(train_file))

	examples_t5_format = []
	labels_t5_format = []
	document_ids = []   # which document the example came from? (e.g, 54f94949-0fb4-45e5-81dd-c4385f681e2b)
	example_ids = []    # which document-type and type-name does the example belong to? (e.g., matriculas.endereco)

	for example in track(examples, description="convert examples to T5 format", disable=False): #not tqdm_enabled):

		# prepare the input
		x = generate_t5_input_sentence(example.context_text, example.question_text, use_sentence_id=True)
		
		# extract answer and start position (squad-example is in evaluate mode)
		y = example.answers[0]['text']  # getting the first answer in the list
		answer_start = example.answers[0]['answer_start']

		# prepate the target
		y = generate_t5_label_sentence(y, answer_start, example.context_text, use_sentence_id=True)
		
		examples_t5_format.append(x)
		labels_t5_format.append(y)
		document_ids.append(example.title)
		example_ids.append(example.qas_id)

	examples_t5_format, labels_t5_format, document_ids, example_ids = balance_data(examples_t5_format,
		labels_t5_format, document_ids, example_ids, negative_ratio=negative_ratio)

	for input, label, document_id, example_id in zip(examples_t5_format, labels_t5_format, document_ids, example_ids):
		if document_id == '93445':
			print(label, example_id)
			print(input[:180])

if __name__ == '__main__':
	main()
