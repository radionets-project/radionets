#########################################################
# Analysis steps
#

all:	mnist_fft\
		mnist_samp\
		calc_normalization

quick_data_and_training: mnist_fft_quick\
						 clean_samp\
						 mnist_samp\
						 calc_normalization\
						 clean_training\
						 cnn_training

#--------------------------------------------------------

mnist_samp:	mnist_samp_train\
			mnist_samp_valid

#--------------------------------------------------------

mnist_fft: ./data/mnist_fft_done
mnist_fft_quick: ./data/mnist_fft_done_quick
mnist_samp_train: ./data/mnist_samp_train.h5
mnist_samp_valid: ./data/mnist_samp_valid.h5
calc_normalization: ./data/normalization_factors.csv
cnn_training: ./training_done
input_plot: ./data/plots/input_done

#########################################################
#	Tasks
#

./data/mnist_fft_done:
		mkdir -p data/
		python create_mnist_fft.py ./../resources/mnist.pkl.gz\
		./data -test False -noise False
		touch ./data/mnist_fft_done

./data/mnist_fft_done_quick:
		mkdir -p data/
		python create_mnist_fft.py ./../resources/mnist.pkl.gz\
		./data -test True -noise False
		touch ./data/mnist_fft_done

./data/mnist_samp_train.h5:
		python create_mnist_fft_sampled.py\
		data/mnist_train.h5 data/mnist_samp_train.h5\
		../sampling/layouts/vlba.txt -train True\
		-specific_mask True\
		-lon -80\
		-lat 50\
		-steps 50

./data/mnist_samp_valid.h5:
		python create_mnist_fft_sampled.py\
		data/mnist_valid.h5 data/mnist_samp_valid.h5\
		../sampling/layouts/vlba.txt -train False\
		-specific_mask True\
		-lon -80\
		-lat 50\
		-steps 50

./data/normalization_factors.csv:
		python calculate_normalization.py\
		-log False -use_mask True\
		data/mnist_samp_train.h5 data/mnist_samp_valid.h5\
		data/normalization_factors.csv

./training_done:
		mkdir -p models/
		python train_cnn.py ./data/mnist_samp_train.h5\
		./data/mnist_samp_valid.h5 ./models/two_channels_2.model\
		cnn ./data/normalization_factors.csv\
		1 1e-3 -log False\
		-pretrained True ./models/two_channels_2.model\
		-inspection True
		touch ./training_done

./data/plots/input_done:
		mkdir -p data/
		mkdir -p data/plots
		python create_input_plot.py ./../resources/mnist.pkl.gz
		touch ./data/plots/input_done

clean:
	rm -rf ./data
	rm -f ./training_done

clean_training:
	rm -f ./training_done

clean_input_plot:
	rm -rf ./data/plots
clean_samp:
	rm -f ./data/*_samp_*.h5