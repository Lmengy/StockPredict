import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class Preprocessing():
	def __init__(self, filename, ndays=7):
		self.filename = filename
		self.ndays = ndays
		self.sc = StandardScaler()

	def preprocess(self):
		stock_def_df = pd.read_csv('{}{}'.format("./data/", self.filename), header=None)
		#stock_def_df = pd.read_csv('{}{}'.format("../data/", self.filename), header=None)
		#self.ndays = 7
		df_proc = DataFrameProcessing(stock_def_df, self.ndays)

		# 開始處理資料
		df_proc.data_proc()
		stock_input_data = df_proc.get_test_data()

		# Problem: 3 維數組無法套用 transform
		# Try to:
		#	1. 尋找其他可用於多維數組的標準化函數
		#	2. 自行編寫標準化函數
		#	3. 當資料在分割前（2 維）先完成標準化
		# 以下採用第 3 個方法
		
		# normalize
		stock_input_data = df_proc.normalize(stock_input_data)

		# convert data from 2d to 3d
		stock_input_data = df_proc.data_reshape(stock_input_data)

		# input_train, target_train, input_test, target_test = self._split_data(stock_input_df, test_size=0.2)
		input_train, target_train, input_test, target_test = self.test_split_data(stock_input_data, test_size=0.2)

		# 		input_train.shape -> (712, 5, 7)
		# 		target_train.shape -> (179, 5, 7)
		# 		input_test.shape -> (712, 7)
		# 		target_test.shape -> (179, 7)
		"""
		print(stock_input_data.shape)
		print(input_train.shape)
		print(input_test.shape)
		print(target_train.shape)
		print(target_test.shape)

		import matplotlib.pyplot as plt
		r = target_test.flatten()
		plt.plot(r)
		plt.show()
		"""

		# for p2330.csv
		# original:
		# 		input_train.shape -> (712, 35)
		# 		target_train.shape -> (712, 7)
		# try to fix:
		# 		input_train.shape -> (712, 7, 5)
		# 		target_train.shape -> (712, 7, 1)

		# TODO: input_train shuffle
		
		return input_train, target_train, input_test, target_test


	def _split_data(self, df, test_size=0.2):
		n = int(df.shape[0] * (1 - test_size))

		input_train = df.iloc[:n].values
		target_train = df.loc[:n, pd.IndexSlice["close", :]].values
		input_test = df.iloc[n:].values
		target_test = df.loc[n:, pd.IndexSlice["close", :]].values

		return input_train, target_train, input_test, target_test
		
	
	def test_split_data(self, data, test_size=0.2):
		n = int(data.shape[0] * (1 - test_size))
		input_train = data[:n, :, :]
		#target_train = data[:n, 0, :]
		target_train = data[1:n+1, 0, :]
		input_test = data[n:, :, :]
		#target_test = data[n:, 0, :]
		target_test = data[n+1:, 0, :]
		return input_train, target_train, input_test, target_test


class DataFrameProcessing():
	def __init__(self, stock_df, ndays):
		self.stock_df = stock_df
		self.data_len = stock_df.shape[0] - 1	# 總資料筆數為 dataframe 長度減去 dataframe 中的 keys field
		self.indicators = list(stock_df.iloc[0, :])	# stock_df 的第一列為 keys field
		self.ndays = ndays
		self.data_len_per_ndays = self.data_len // self.ndays
		self.data = None
		self.sc = StandardScaler()


	def data_proc(self):

		# 將輸入資料分為 m 組，每組包含 n*k 筆資料的 DataFrame（n 表示每組資料包含的日期天數，k 是待預測的指標數量）
		# 不滿足 n 日的資料組須捨棄
		# 以 2330.csv 為例，若將資料分組為 7*5 (7 days * (open, close, low, high, volume))
		# 由於總資料筆數為 6239，每 7 筆資料一組則最後會多出 2 筆資料，因此不採用最後 2 筆資料
		idxes_to_drop = self.data_len % self.ndays
		if idxes_to_drop > 0:
			self.stock_df = self.stock_df.drop(
								self.stock_df.tail(idxes_to_drop).index)
			self.data_len -= idxes_to_drop
		
		# 建立一個 np.array 類型的 data 以便於資料形狀的縮放
		self.data = np.array(self.stock_df)

		# 以 dataframe 直接轉換成 np.array 後，由於 np.array 會一併將 keys field 加入陣列中，因此必須將第一列移除
		self.data = np.delete(self.data, 0, axis=0)

		# convert data type from 'str' to 'float64'
		self.data = self.data.astype('float64')


	# 此函數僅作為測試
	# 為確保模型能正確的輸入資料，此函數將回傳欲測試的輸入資料
	def get_test_data(self):
		return self.data


	def create_df(self, df_type="input"):
		_df = None
		col_series = None
		idx_series = None

		if df_type == "input":
			# index 和 column 以 Series 為對象，在 MultiIndex 下必須以 tuple 來索引
			# 因此需要的 Series sizes 為 n*k*len(tuple) => n*k*2
			stocklbl_col = [i % self.ndays + 1 for i in range(self.data.shape[1] * self.data.shape[2])]
			stocklbl_sec_col = [self.indicators[i // self.ndays] for i in range(self.data.shape[1] * self.data.shape[2])]

			# input shape 為 (m, k*n)
			col_series = [np.array(stocklbl_sec_col), np.array(stocklbl_col)]
			idx_series = [i + 1 for i in range(self.data_len_per_ndays)]

		elif df_type == "output":
			# Series sizes 為 m*n*len(tuple) => m*n*2
			nday_idx = [i % self.ndays + 1 for i in range(self.data.shape[0] * self.data.shape[2])]
			nday_sec_idx = [i // self.ndays + 1 for i in range(self.data.shape[0] * self.data.shape[2])]

			# input shape 為 (m*n, k)
			idx_series = [np.array(nday_sec_idx), np.array(nday_idx)]
			col_series = self.indicators

		_df = pd.DataFrame(
						self._data_reshape(df_type),
						columns = col_series, 
						index = idx_series
						)
		return _df

	
	def normalize(self, input_data):
		self.sc.fit(input_data)
		return self.sc.transform(input_data)
	

	# FIXME: can't denormalize the target in shape(712, 7)
	def denormalize(self, input_data):
		return self.sc.inverse_transform(input_data)
	
	
	def data_reshape(self, input_data):

		# 將陣列從時間序索引轉換為標籤序索引，並將每組標籤內的數值按照 self.ndays 劃分
		# 最後 self.data.shape 為 (m, k, n)
		input_data = input_data.T
		input_data = np.array(np.split(
								input_data, 
								self.data_len_per_ndays, 
								axis=1
								))
		return input_data

		"""
		self.data = self.data.T
		self.data = np.array(np.split(
								self.data, 
								self.data_len_per_ndays, 
								axis=1
								))
		"""


	"""
	# 將 data shape 轉換為 input dataframe shape 或 output dataframe shape ，預設為 input dataframe shape
	# 輸入參數僅 input 和 output，其餘參數則返回 None
	def _data_reshape(self, df_type="input"):
		_data = None

		# shape (m, k, n) to shape (m, k*n)
		if df_type == "input":
			_data = np.array([np.concatenate(self.data[i], axis=0) for i in range(self.data.shape[0])])
		# shape (m, k, n) to shape (m*n, k)
		elif df_type == "output":
			_data = np.concatenate(self.data, axis=1).T

		return _data
	"""

if __name__ == "__main__":
	a = Preprocessing("p2330.csv")
	a.preprocess()
	# b = DataFrameProcessing(pd.read_csv('../data/p2330.csv', header=None), 7)
