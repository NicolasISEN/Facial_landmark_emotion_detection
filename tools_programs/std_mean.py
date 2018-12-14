import h5py
import numpy as np
import menpo.image as mi
import menpo.io as mio

debug = False

def createMeanAndStd(filenames,meanname,stdname):
	global debug
	#mean_image = np.zeros((1,110,110), dtype=np.float32)
	#std_image = np.zeros((1,110,110), dtype=np.float32)
	sum_mean_images = np.zeros((1,128,128), dtype=np.float32)
	sum_std_images = np.zeros((1,128,128), dtype=np.float32)
	len_total_data_images = 0
	print("Loading mean ...")
	for filename in filenames:
		print(filename)
		f = h5py.File(filename, 'r')

		# Get the data
		data = {}
		for key in list(f.keys()):
			data[key] = np.asarray(list(f[key]))
	
		if debug:
			print()

		for image in data["images"]:
			sum_mean_images = sum_mean_images + image
		len_total_data_images += len(data["images"])
	mean_images = np.array(sum_mean_images)/len_total_data_images
	del sum_mean_images
	print("Done")
	
	print("Loading std ...")
	for filename in filenames:
		print(filename)
		f = h5py.File(filename, 'r')

		# Get the data
		data = {}
		for key in list(f.keys()):
			data[key] = np.asarray(list(f[key]))
	
		if debug:
			print(len(images))

		for image in data["images"]:
			sum_std_images = sum_std_images + np.power((image-mean_images), 2)
	std_images = np.array(sum_std_images)/len_total_data_images
	std_images = np.sqrt(std_images)
	del sum_std_images
	print("Done")

	print("Exporting...")
	mio.export_image(mi.Image(mean_images), meanname)

	mio.export_image(mi.Image(std_images), stdname)
	print("Done")

def createMeanAndStd2(filename,meanname,stdname):
	global debug
	#mean_image = np.zeros((1,110,110), dtype=np.float32)
	#std_image = np.zeros((1,110,110), dtype=np.float32)
	sum_mean_images = np.zeros((1,128,128), dtype=np.float32)
	sum_std_images = np.zeros((1,128,128), dtype=np.float32)
	len_total_data_images = 0
	print("Loading mean ...")
	print(filename)
	f = h5py.File(filename, 'r')

	# Get the data
	data = {}
	for key in list(f.keys()):
		data[key] = np.asarray(list(f[key]))
	
	if debug:
		print()

	for image in data["images"]:
		sum_mean_images = sum_mean_images + image
	len_total_data_images += len(data["images"])
	mean_images = np.array(sum_mean_images)/len_total_data_images
	del sum_mean_images
	print("Done")
	
	print("Loading std ...")
	print(filename)
	f = h5py.File(filename, 'r')

	# Get the data
	data = {}
	for key in list(f.keys()):
		data[key] = np.asarray(list(f[key]))
	
	if debug:
		print(len(images))

	for image in data["images"]:
		sum_std_images = sum_std_images + np.power((image-mean_images), 2)
	std_images = np.array(sum_std_images)/len_total_data_images
	std_images = np.sqrt(std_images)
	del sum_std_images
	print("Done")

	print("Exporting...")
	mio.export_image(mi.Image(mean_images), meanname)

	mio.export_image(mi.Image(std_images), stdname)
	print("Done")

if __name__ == "__main__":

	training_filenames = ["training_dataset_pack0.h5","training_dataset_pack1.h5","training_dataset_pack2.h5","training_dataset_pack3.h5"]
	mean_training_filename = "mean_training.png" 
	std_training_filename = "std_training.png"
	validation_filename = "validation_dataset_pack0.h5"
	mean_validation_filename = "mean_validation.png" 
	std_validation_filename = "std_validation.png"
	test_filename = "test_dataset_pack0.h5"
	mean_test_filename = "mean_test.png" 
	std_test_filename = "std_test.png"
	
	createMeanAndStd(training_filenames,mean_training_filename,std_training_filename)
	createMeanAndStd2(validation_filename,mean_validation_filename,std_validation_filename)
	createMeanAndStd2(test_filename,mean_test_filename,std_test_filename)