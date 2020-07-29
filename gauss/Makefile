#########################################################
# Analysis steps
#

all:	gauss_fft\
		clean_samp\
		gauss_samp\
		calc_normalization\
		clean_training\
		cnn_training

quick_data_and_training: gauss_fft_quick\
						 clean_samp\
						 gauss_samp\
						 calc_normalization\
						 clean_training\
						 cnn_training

training: calc_normalization\
                                                 clean_training\
                                                 cnn_training

data:   gauss_fft\
                clean_samp\
                gauss_samp

testdata: gauss_test
#--------------------------------------------------------

gauss_samp:	gauss_samp_train\
				gauss_samp_valid
gauss_test:     gauss_testset\
                                test_normalization

#--------------------------------------------------------

gauss_fft: ./data/gauss_fft_done
gauss_fft_quick: ./data/gauss_fft_done_quick
gauss_samp_train: ./data/gauss_samp_train.h5
gauss_samp_valid: ./data/gauss_samp_valid.h5
calc_normalization: ./data/normalization_factors.csv
cnn_training: ./training_done
input_plot: ./data/plots/input_done
find_lr: ./lr_find
gauss_testset: ./test_data/gauss_testset
test_normalization: ./test_data/normalization_factors.csv
#########################################################
#	Tasks
#

./data/gauss_fft_done:
		mkdir -p data/
		python create_gauss_fft.py ./data\
		50000 15 False\
		-test False -noise False
		touch ./data/gauss_fft_done

./data/gauss_fft_done_quick:
		mkdir -p data/
		python create_gauss_fft.py ./data\
		500 10 True\
		-test True -noise False
		touch ./data/gauss_fft_done

./data/gauss_samp_train.h5:
		python create_gauss_fft_sampled.py\
		data/gauss_train.h5 data/gauss_samp_train.h5\
		../simulations/layouts/vlba.txt -train True\
		-specific_mask True\
		-lon -80\
		-lat 50\
		-steps 50

./data/gauss_samp_valid.h5:
		python create_gauss_fft_sampled.py\
		data/gauss_valid.h5 data/gauss_samp_valid.h5\
		../simulations/layouts/vlba.txt -train False\
		-specific_mask True\
		-lon -80\
		-lat 50\
		-steps 50

./data/normalization_factors.csv:
		python calculate_normalization.py\
		data/gauss_samp_train.h5 \
		data/normalization_factors.csv

./training_done:
		mkdir -p models/
		python train_cnn.py ./data/gauss_samp_train.h5\
		./data/gauss_samp_valid.h5 ./models/unet_gauss.model\
		UNet ./data/normalization_factors.csv\
		1000 2e-5 feature_loss -log False\
		-pretrained False \
		-inspection False
		touch ./training_done

./data/plots/input_done:
		mkdir -p data/
		mkdir -p data/plots
		python create_input_plot.py ./../resources/mnist.pkl.gz
		touch ./data/plots/input_done

./lr_find:
		mkdir -p models/
		python find_lr.py ./data/gauss_samp_train.h5\
                ./data/gauss_samp_valid.h5 ./models/unet_gauss.model\
                UNet feature_loss ./data/normalization_factors.csv -log False\
                -pretrained False ./models/unet_gauss.model\
                -save True

./test_data/gauss_testset:
		mkdir -p test_data/
		python create_gauss_testset.py ./test_data/gauss_samp_test.h5\
                12500 15 False \
                ../simulations/layouts/vlba.txt\
		-noise False -specific_mask True \
                -lon -80\
                -lat 50\
                -steps 50

###########################################################################

clean:
	rm -rf ./data
	rm -rf ./test_data
	rm -f ./training_done

clean_training:
	rm -f ./training_done

clean_input_plot:
	rm -rf ./data/plots
clean_samp:
	rm -f ./data/*_samp_*.h5

clean_test:
	rm -rf ./test_data
