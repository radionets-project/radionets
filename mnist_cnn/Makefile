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
		./data -size 63 -noise False -bundle_size 500
		# touch ./data/mnist_fft_done

./data/mnist_samp_done:
		python scripts/create_fft_sampled.py \
		./data/ ./data/ \
		../simulations/layouts/vlba.txt \
		-fourier False \
		-specific_mask True \
		-lon -80 \
		-lat 50 \
		-steps 50
		# touch ./data/mnist_samp_done

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
		python plot_loss.py\
		${path_model}\
		${arch}\

./lr_find:
		mkdir -p models/
		python scripts/find_lr.py ${data_path}\
		${arch} ${lr_type} ${norm_path} -fourier ${fourier}\
		-pretrained ${pretrained} ${pre_path}\
		-save ${save}

clean:
	rm -rf ./data
	rm -f ./training_done

clean_training:
	rm -f ./training_done

clean_input_plot:
	rm -rf ./data/plots
clean_samp:
	rm -f ./data/*_samp_*.h5