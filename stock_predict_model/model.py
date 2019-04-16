from keras.models import Sequential,load_model
from keras.layers import Dense, Activation, LSTM, RepeatVector
import matplotlib.pyplot as plt


class StockPredict():
	def __init__(self, ndays, input_train, input_test,target_train, target_test):
		self.ndays = ndays
		self.input_train = input_train
		self.input_test = input_test
		self.target_train = target_train
		self.target_test = target_test
		self.pred_train = None
		self.pred_test = None
		#self.build_model()
		self.run_model()


	def run_model(self):
		model = load_model('./model/test_model_1-1.h5')
		#model = load_model('./model/test_model_2.h5')
		#model = load_model('./model/test_model_3.h5')
		#model = self.build_model()

		target_pred_train = model.predict(self.input_train)
		target_pred_test = model.predict(self.input_test)

		self.pred_train = target_pred_train
		self.pred_test = target_pred_test


	def build_model(self):
		hidden_size = 32
		input_size = (self.input_train.shape[1], self.input_train.shape[2])
		seq_len = self.input_train.shape[0]

		
		model = Sequential([
					LSTM(units=hidden_size, input_shape=input_size, return_sequences=False, activation='relu'),
					Dense(units=self.ndays),
					RepeatVector(seq_len),
					LSTM(units=hidden_size, return_sequences=False),
					Dense(units=self.ndays)
		])

		"""
		model = Sequential([
					LSTM(units=hidden_size, input_shape=input_size, activation='relu'),
					Dense(units=self.ndays) 
		])
		"""

		model.compile(loss='mse', optimizer='adam', metrics=['acc'])
		model.summary()
		model.fit(self.input_train, self.target_train, epochs=1000, batch_size=32) #batch_size=7)
		
		# save model 
		model.save('./model/test_model_3.h5')

		"""
		target_pred_train = model.predict(self.input_train)
		target_pred_test = model.predict(self.input_test)
		
		print(target_pred_train)
		print(target_pred_test)
		print(type(target_pred_train))
		print(type(target_pred_test))
		print(target_pred_train.shape)
		print(target_pred_test.shape)

		self.pred_train = target_pred_train
		self.pred_test = target_pred_test
		"""

		return model


	def get_result(self):
		return self.pred_train, self.pred_test

if __name__ == "__main__":
	main()
