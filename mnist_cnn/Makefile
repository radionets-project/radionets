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
		python create_mnist_fft.py ./../resources/mnist.pkl.gz\
		./data -size 63 -noise True
		# touch ./data/mnist_fft_done

./data/mnist_samp_done:
		python create_fft_sampled.py \
		./data/ ./data/ \
		../simulations/layouts/vlba.txt \
		-fourier False \
		-specific_mask True \
		-lon -80 \
		-lat 50 \
		-steps 50
		# touch ./data/mnist_samp_done

./data/normalization_factors.csv:
		python calculate_normalization.py \
		./data \
		./data/normalization_factors.csv

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
		python find_lr.py ${samp_train}\
		${samp_valid}\
		${arch} ${lr_type} ${norm_path} -log ${log}\
		-pretrained ${pretrained} ${pre_path}\
		-save ${save}

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