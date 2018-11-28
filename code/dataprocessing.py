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
from math import isnan
import pandas as pd
import numpy as np
import csv, re, zipfile

class DataReader:
	"""
	Class to read in, clean up and possibly shuffle and scale different data files.
	Returns two data sets, the original data split into features and labels.
	"""
	def __init__(self, datafile, regression=False):
		self.datafile = datafile
		self.headers = np.zeros(len(pd.read_csv(self.datafile).values[:,0]))
		self.data = []
		self.encoders = []
		self.label_encoder_ = []
		self.labels=[]
		self.features=[]
		self.class_column = -1
		self.regression = regression

	"""
	Reading in data, checking if there are headers, if not, add column names.
	Set data and headers for DataReader object.
	"""
	def read_file(self):
		if zipfile.is_zipfile(self.datafile):
			f = zipfile.ZipFile(self.datafile)
			for name in f.namelist():
				if name.endswith('.txt') or name.endswith('.csv'):
					data = pd.read_csv(f.open(name), dtype={'DOB':object,'NOEXCH':object})
		else:
			sniff = csv.Sniffer()
			sample_bytes = 64
			header_ = sniff.has_header(open(self.datafile).read(sample_bytes))
			"""
			cover_headers = ["elevation", "aspect", "slope", "horizontal_distance_hydrology", "vertical_distance_hydrology",
				"horizontal_distance_roadways", "hillshade_9am", "hillshade_noon", "hillshade_3pm", "wilderness_rawah",
				"wilderness_neota", "wilderness_comanche", "wilderness_cache", "ELU_2702", "ELU_2703", "ELU_2704", "ELU_2705",
				"ELU_2706", "ELU_2717", "ELU_3501", "ELU_3502", "ELU_4201", "ELU_4703", "ELU_4704", "ELU_4744", "ELU_4758",
				"ELU_5101", "ELU_5151", "ELU_6101", "ELU_6102", "ELU_6731", "ELU_7101", "ELU_7102", "ELU_7103", "ELU_7201",
				"ELU_7202", "ELU_7709", "ELU_7710", "ELU_7745", "ELU_7746", "ELU_7755", "ELU_7756", "ELU_7757", "ELU_7790",
				"ELU_8703", "ELU_8707", "ELU_8708", "ELU_8771", "ELU_8772", "ELU_8876", "cover_class"]

			nursery_headers = ["parents","has_nurs","form","children","housing","finance","social","health"]
			"""

			if not header_:
				"""
				if "cov" in self.datafile:
					data = pd.read_csv(self.datafile, header=None, names=cover_headers)
				if "nurse" in self.datafile:
					data = pd.read_csv(self.datafile, header=None, names=nursery_headers)
				else:
				"""
				data = pd.read_csv(self.datafile, header=None)
				self.headers = data.columns.values
			else:
				data = pd.read_csv(self.datafile)
		if self.regression:
			data.iloc[:,self.class_column] = pd.qcut(data.iloc[:,self.class_column], 4)
		self.data = data.values

	"""
	Make sure each class has at least 10 instances for stratified 10-fold CV.
	"""
	def check_class_instances(self):
		unique, counts = np.unique(self.labels, return_counts=True)
		for i,c in enumerate(counts):
			if c < 10:
				value = unique[i]
				print(i, value)
				self.features = np.asarray([row for i,row in enumerate(self.features) if self.labels[i] != value])
				self.labels = np.asarray([l for l in self.labels if l != value])

	"""
	Check if any feature columns or labels are non-numerical, if so, convert to numbers.
	"""
	def categorical_to_num(self):
		numeric_data_types = set('buifc')
		transformed = []
		#Transform features that are non numeric
		for i,col in enumerate(self.features.T):
			conv = pd.Series(col)
			conv = pd.to_numeric(conv, errors="ignore")
			if not conv.dtype.kind in numeric_data_types:
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
	Check for missing values.
	If the number of data points with missing values is less than 10% of the dataset, 
	remove the rows, otherwise, remove the columns.
	"""
	def remove_missing_values(self):
		attribute_indeces = []
		datapoint_indices = []

		for i, col in enumerate(self.features.T):
			if "?" in col or "" in col:
				attribute_indeces.append(i)
		for i, row in enumerate(self.features):
			if "?" or "" in row:
				datapoint_indices.append(i)
		if len(datapoint_indices)/self.features.shape[0] < 0.1:
			print("deleting rows")
			self.features = np.delete(self.features, datapoint_indices, axis=0)
			self.labels = np.delete(self.labels, datapoint_indices, axis=0)
		else:
			print("deleting columns")
			self.features = np.delete(self.features, attribute_indeces, axis=1)

	"""
	Split data into features and labels, based on class_column parameter.
	"""
	def split_feat_labels(self, class_column):
		self.labels = self.data[:, class_column]
		self.features = np.delete(self.data, class_column, axis=1)

	"""
	Function to run all other datareader functionality.
	Parameters:
	class_column: which column to use for classification goal (-1 if last column)
	bool_scale: boolean to tell if data needs to be scaled
	bool_shuffled: boolean to tell if data needs to be shuffled
	"""
	def run(self, class_column=-1, bool_scale=False, bool_shuffled=False):
		self.class_column = class_column
		self.read_file()
		if bool_shuffled:
			np.random.shuffle(self.data)
		self.split_feat_labels(self.class_column)
		self.remove_missing_values()
		self.check_class_instances()
		self.categorical_to_num()
		if bool_scale:
			self.features = preprocessing.scale(self.features)
		return (self.features, self.labels)

if __name__ == '__main__':
	dr = DataReader("../data/nursery.csv")
	dr.run()
