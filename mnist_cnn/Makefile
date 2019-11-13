#########################################################
# Analysis steps
#

all:	mnist_fft\
		mnist_samp

quick_data_and_training: mnist_fft_quick\
						 mnist_samp\
						 cnn_training

#--------------------------------------------------------

mnist_samp:	mnist_samp_train\
			mnist_samp_valid

#--------------------------------------------------------

mnist_fft: ./data/mnist_fft_done
mnist_fft_quick: ./data/mnist_fft_done_quick
mnist_samp_train: ./data/mnist_samp_train.h5
mnist_samp_valid: ./data/mnist_samp_valid.h5
cnn_training: ./training_done

#########################################################
#	Tasks
#

./data/mnist_fft_done:
		mkdir -p data/
		python create_mnist_fft.py ./../resources/mnist.pkl.gz\
		./data -test False
		touch ./data/mnist_fft_done

./data/mnist_fft_done_quick:
		mkdir -p data/
		python create_mnist_fft.py ./../resources/mnist.pkl.gz\
		./data -test True
		touch ./data/mnist_fft_done

./data/mnist_samp_train.h5:
		python create_mnist_fft_sampled.py\
		data/mnist_train.h5 data/mnist_samp_train.h5\
		../sampling/layouts/vlba.txt -train True

./data/mnist_samp_valid.h5:
		python create_mnist_fft_sampled.py\
		data/mnist_valid.h5 data/mnist_samp_valid.h5\
		../sampling/layouts/vlba.txt -train False 

./training_done:
		mkdir -p models/
		python train_cnn.py ./data/mnist_samp_train.h5\
		./data/mnist_samp_valid.h5 ./models/cuda_test.model\
		10 5e-1 -log True -mask True\
		-pretrained False\
		-inspection True
		touch ./training_done

clean:
	rm -rf ./data
	rm -f ./training_done
clean_training:
	rm -f ./training_done

