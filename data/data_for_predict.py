import pandas as pd
import numpy as np

def main():
	df = pd.read_csv("./2330.csv", header=None)
	new_df_dict = {	"open":		df.iloc[:, 3],
					"close":	df.iloc[:, 6],
					"high":		df.iloc[:, 4],
					"low":		df.iloc[:, 5],
					"volume":	df.iloc[:, 1],
	}
	"""
	names = new_df_dict.keys()
	firstKey = 	names[0]
	formats = ['f8']*len(names)
	dtype = dict(names = names, formats = formats)
	
	values = [tuple(new_df_dict[k][0] for k in new_df_dict.keys())]
	data = np.array(values, dtype = dtype)
	for i in range(1, len(new_df_dict[names])):
		values = [tuple(new_df_dict[k][i] for k in new_df_dict.keys())]
		data_tmp = np.array(values, dtype = dtype)
		data = np.concatenate((data, data_tmp))
	"""

	new_df = pd.DataFrame(new_df_dict)

	# items() 方法返回的是包含多組 (key, val) 形式的 list
	# data_array = np.array((key, val) for (key, val) in new_df_dict.items(), dtype) 
	#data_array = np.array(list(new_df_dict.items()), dtype)
	# data_array = np.fromiter(new_df_dict.items(), dtype = f, count=len(new_df_dict))
	
	new_df.to_csv('p2330.csv', index=False)
	#ndf = np.array(new_df)
	
	#print(new_df.head())
	#print(ndf)

if __name__ == "__main__":
	main()
