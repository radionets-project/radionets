#########################################################
# Analysis steps
#
-include .makerc
all:	mnist_fft\
		mnist_samp\
		calc_normalization\
		clean_training\
		cnn_training\
		input_plot

quick_data_and_training: mnist_fft_quick\
						 clean_samp\
						 mnist_samp\
						 calc_normalization\
						 clean_training\
						 cnn_training

training: calc_normalization\
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
find_lr: ./lr_find

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
		python calculate_normalization.py \
		data/mnist_samp_train.h5 \
		data/normalization_factors.csv

./training_done:
		mkdir -p ${path}
		python train_cnn.py ${samp_train}\
		${samp_valid} ${path_model}\
		${arch} ${norm_path}\
		${epochs} ${lr} ${lr_type} -log ${log}\
		-pretrained ${pretrained} ${pre_path}\
		-inspection ${inspection}
		# touch ./training_done

./plot_loss:
		mkdir -p ${path}
		python plot_loss.py\
		${path_model}\
		${arch}\

./lr_find:
		mkdir -p models/
<<<<<<< HEAD
		python train_cnn.py ./data/mnist_samp_train.h5\
		./data/mnist_samp_valid.h5 ./models/unet_mnist.model\
		UNet ./data/normalization_factors.csv\
		50 1e-3 -log False\
		-pretrained False ./models/autoencoder_mnist.model \
		-inspection False
		touch ./training_done
||||||| merged common ancestors
		python train_cnn.py ./data/mnist_samp_train.h5\
		./data/mnist_samp_valid.h5 ./models/test_pred.model\
		cnn ./data/normalization_factors.csv\
		150 1e-3 -log False\
		-pretrained False ./models/test.model\
		-inspection True
		touch ./training_done
=======
		python find_lr.py ${samp_train}\
		${samp_valid}\
		${arch} ${lr_type} ${norm_path} -log ${log}\
		-pretrained ${pretrained} ${pre_path}\
		-save ${save}
>>>>>>> origin/master

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
