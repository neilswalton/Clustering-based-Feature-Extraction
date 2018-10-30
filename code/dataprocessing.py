################################################################
#                                                              #
# Dataprocessing class to read in the different data sets.     #
# Datasets that are being read in are crime and communities,   #
# (promotional mail response and Dota game statistics)         #
#                                                              #
# Authors: Amy Peerlinck and Neil Walton                       #
#                                                              #
#                                                              #
################################################################

from sklearn import preprocessing, feature_extraction
import pandas as pd
import numpy as np
import csv

class DataReader:
	"""
	Class to read in, clean up and possibly shuffle and scale different data files.
	Returns two data sets, the original data split into features and labels.
	"""
	def __init__(self, datafile):
		self.datafile = datafile
		self.headers = np.zeros(len(pd.read_csv(self.datafile).values[:,0]))
		self.data = []
		self.encoders = []
		self.label_encoder_ = []
		self.labels=[]
		self.features=[]

	"""
	Reading in data, checking if there are headers, if not, add column names.
	Set data and headers for DataReader object. 
	"""
	def read_file(self):
		sniff = csv.Sniffer()
		sample_bytes = 64
		header_ = sniff.has_header(open(self.datafile).read(sample_bytes))

		cover_headers = ["elevation", "aspect", "slope", "horizontal_distance_hydrology", "vertical_distance_hydrology", 
			"horizontal_distance_roadways", "hillshade_9am", "hillshade_noon", "hillshade_3pm", "wilderness_rawah", 
			"wilderness_neota", "wilderness_comanche", "wilderness_cache", "ELU_2702", "ELU_2703", "ELU_2704", "ELU_2705", 
			"ELU_2706", "ELU_2717", "ELU_3501", "ELU_3502", "ELU_4201", "ELU_4703", "ELU_4704", "ELU_4744", "ELU_4758",
			"ELU_5101", "ELU_5151", "ELU_6101", "ELU_6102", "ELU_6731", "ELU_7101", "ELU_7102", "ELU_7103", "ELU_7201", 
			"ELU_7202", "ELU_7709", "ELU_7710", "ELU_7745", "ELU_7746", "ELU_7755", "ELU_7756", "ELU_7757", "ELU_7790",
			"ELU_8703", "ELU_8707", "ELU_8708", "ELU_8771", "ELU_8772", "ELU_8876", "cover_class"]

		nursery_headers = ["parents","has_nurs","form","children","housing","finance","social","health"]

		if not header_:
			if self.datafile.contains("cov"):
				data = pd.read_csv(self.datafile, names=cover_headers)
			if self.datafile.contains("nurse"):
				data = pd.read_csv(self.datafile, names=nursery_headers)
		else:
			data = pd.read_csv(self.datafile)
		self.headers = data.columns.values
		self.data = data.values

	"""

	"""
	def check_class_instances(self):
		print(self.data.groupby(self.headers[-1]).nunique())


	"""
	Check if any feature columns or labels are non-numerical, if so, convert to numbers.
	"""
	def categorical_to_num(self):
		numeric_data_types = set('buifc')
		transformed = []
		# Transform features that are non numeric
		for i,col in enumerate(self.features.T):
			if not col.dtype.kind in numeric_data_types:
				encoder = preprocessing.LabelEncoder()
				encoder.fit(self.features[:,i])
				self.encoders.append(encoder)
				transformed.append(encoder.transform(self.features[:,i]))
		for i, col in enumerate(transformed):
			self.features[:,i] = col
		#Transform labels
		if not self.labels.dtype.kind in numeric_data_types:
			self.label_encoder_ = preprocessing.LabelEncoder()
			self.label_encoder_.fit(self.labels)
			self.labels = self.label_encoder_.transform(self.labels)

	"""
	Convert data back to original format.
	Keeps DataReader object feature and label instances in numerical format.
	"""
	def num_to_category(self):
		numeric_data_types = set('buifc')
		counter = 0
		original = []
		for i,col in enumerate(self.features.T):
			if not col.dtype.kind in numeric_data_types:
				encoder = self.encoders[counter]
				original.append(encoder.inverse_transform(self.features[:,i].astype(int)))
				counter+=1
		original_labels = self.label_encoder_.inverse_transform(self.labels.astype(int))
		return np.asarray(original), original_labels


	"""
	Split data into features and labels, based on class_column parameter.
	"""
	def split_feat_labels(self, class_column):
		self.labels = self.data[:, class_column]
		self.features = np.delete(self.data, class_column, 1)

	"""
	Function to run all other datareader functionality. 
	Parameters:
	class_column: which column to use for classification goal (-1 if last column)
	bool_scale: boolean to tell if data needs to be scaled
	bool_shuffled: boolean to tell if data needs to be shuffled
	"""
	def run(self, class_column=-1, bool_scale=False, bool_shuffled=False):
		self.read_file()
		self.check_class_instances()
		if bool_shuffled:
			np.random.shuffle(self.data)
		self.split_feat_labels(class_column)
		self.categorical_to_num()
		if bool_scale:
			self.features = preprocessing.scale(self.features)
		return self.features, self.labels


dr = DataReader("../data/nursery.csv")
print(dr.run())
print(dr.num_to_category())