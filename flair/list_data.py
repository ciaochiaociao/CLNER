from .data import *
class ListCorpus(Corpus):
	def __init__(
		self,
		train: List[FlairDataset],
		dev: List[FlairDataset],
		test: List[FlairDataset],
		name: str = "listcorpus",
		targets: list = [],
	):
		# In this Corpus, we set train list to be our target to train, we keep self._train the same as the Class Corpus as the counting and preprocessing is needed
		self.train_list: List[FlairDataset] = train
		self.dev_list: List[FlairDataset] = dev
		self.test_list: List[FlairDataset] = test
		self._train: FlairDataset = ConcatDataset([data for data in train])
		self._dev: FlairDataset = ConcatDataset([data for data in dev])
		self._test: FlairDataset = ConcatDataset([data for data in test])
		self.name: str = name
		self.targets = targets

	def downsample(self, percentage: float = 0.1, only_downsample_train=False):
		for dataset_split in ('train', 'dev', 'test'):
			dataset_list_name = dataset_split + '_list'
			dataset_list = getattr(self, dataset_list_name)
			if only_downsample_train:
				if dataset_split != 'train':
					continue
			new_list = [self._downsample_to_proportion(dataset, percentage) for dataset in dataset_list]
			setattr(self, dataset_list_name, new_list)
			setattr(self, '_' + dataset_split, ConcatDataset([data for data in new_list]))
		return self
