#########################################################
# Analysis steps
#
-include .makerc

dataset:	mnist_fft\
			mnist_samp\
			calc_normalization

#--------------------------------------------------------

mnist_fft: ./data/mnist_fft_done
mnist_samp: ./data/mnist_samp_done
calc_normalization: ./data/normalization_factors.csv
cnn_training: ./training_done
input_plot: ./data/plots/input_done
find_lr: ./lr_find

#########################################################
#	Tasks
#

./data/mnist_fft_done:
		mkdir -p data/
		python scripts/create_mnist_fft.py ./../resources/mnist.pkl.gz\
		./data -size 64 -noise False -bundle_size 500
		touch ./data/mnist_fft_done

./data/mnist_samp_done:
		python scripts/create_fft_sampled.py \
		./data/ ./data/ \
		../simulations/layouts/vlba.txt \
		-fourier False \
		-real_imag True\
		-specific_mask True \
		-lon -80 \
		-lat 50 \
		-steps 50
		touch ./data/mnist_samp_done

./data/normalization_factors.csv:
		python scripts/calculate_normalization.py \
		./data \
		./data/normalization_factors.csv

./training_done:
		mkdir -p ${path}
		python scripts/train_cnn.py ${data_path}\
		${path_model}\
		${arch} ${norm_path}\
		${epochs} ${lr} ${lr_type} ${bs} -fourier ${fourier}\
		-pretrained ${pretrained} ${pre_path}\
		-inspection ${inspection}
		# touch ./training_done

./plot_loss:
		mkdir -p ${path}
		python scripts/plot_loss.py\
		${path_model}\
		${arch}\

./lr_find:
		mkdir -p ${path}
		python scripts/find_lr.py ${data_path}\
		${arch} ${path} ${lr_type} ${norm_path}\
		-max_iter ${max_iter} -min_lr ${min_lr}\
		-max_lr ${max_lr} -fourier ${fourier}\
		-pretrained ${pretrained} ${pre_path}\
		-save ${save} -test ${test}

./visualize_predictions:
		python scripts/visualize_predictions.py\
		${data_path} ${arch} ${path_model}\
		${norm_path} -num_images ${num_images}

./visualize_dataset:
		python scripts/visualize_preprocessing.py\
		${data_path}\
		-num_images ${num_images}

clean:
	rm -rf ./data
	rm -f ./training_done

clean_training:
	rm -f ./training_done

clean_input_plot:
	rm -rf ./data/plots
clean_samp:
	rm -f ./data/*_samp_*.h5
