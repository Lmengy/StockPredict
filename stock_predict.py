import sys
import getopt
from data_processing import data_processing
from stock_predict_model import model

def main():
	#opts, args = getopt.getopt(sys.argv[1:], "")
	#stock_symbol = args[0]
	stock_symbol = 2330
	datafile = 'p{}{}'.format(stock_symbol, ".csv")
	#print(datafile)
	# TODO: input ndays
	# processing the stock data
	stock_proc = data_processing.Preprocessing(datafile, ndays=7)
	input_train, target_train, input_test, target_test = stock_proc.preprocess()

	# TODO: input data to predict model 
	s = model.StockPredict(7, input_train, input_test, target_train, target_test)
	result_train, result_test = s.get_result()
	#result_train, result_test = s.destandardize(result_train, result_test)

	
	# TODO: output data visualization
	f_target_test = target_test.flatten()
	f_result_test = result_test.flatten()

	import matplotlib.pyplot as plt
	plt.figure(1)
	plt.xlabel('days')
	plt.ylabel('close')
	plt.plot(f_target_test[:712], color='red')
	plt.plot(f_result_test[:712], color='blue')
	
	plt.figure(2)
	plt.xlabel('days')
	plt.ylabel('close')
	plt.plot(f_target_test[712:], color='red')
	plt.plot(f_result_test[712:], color='blue')
	plt.show()


if __name__ == "__main__":
	main()
