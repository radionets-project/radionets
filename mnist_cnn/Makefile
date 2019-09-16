#########################################################
# Analysis steps
#

all:	mnist_fft\
		mnist_samp

#--------------------------------------------------------

mnist_samp:	mnist_samp_train\
			mnist_samp_valid

#--------------------------------------------------------

mnist_fft: ./data/mnist_fft_done
mnist_samp_train: ./data/mnist_samp_train.h5
mnist_samp_valid: ./data/mnist_samp_valid.h5

#########################################################
#	Tasks
#

./data/mnist_fft_done:
		python create_mnist_fft.py ./data/mnist/mnist.pkl.gz\
		./data
		touch ./data/mnist_fft_done

./data/mnist_samp_train.h5:
		python create_mnist_fft_sampled.py\
		data/mnist_train.h5 data/mnist_samp_train.h5\
		sampling/vlba.txt -train True

./data/mnist_samp_valid.h5:
		python create_mnist_fft_sampled.py\
		data/mnist_valid.h5 data/mnist_samp_valid.h5\
		sampling/vlba.txt -train False 