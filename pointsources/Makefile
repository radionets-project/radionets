#########################################################
# Analysis steps
#

all:	pointsources_fft\
		clean_samp\
		pointsources_samp\
		calc_normalization\
		clean_training\
		cnn_training

quick_data_and_training: pointsources_fft_quick\
						 clean_samp\
						 pointsources_samp\
						 calc_normalization\
						 clean_training\
						 cnn_training

training: calc_normalization\
                                                 clean_training\
                                                 cnn_training

data:   pointsources_fft\
                clean_samp\
                pointsources_samp

testdata: pointsources_test
#--------------------------------------------------------

pointsources_samp:	pointsources_samp_train\
				pointsources_samp_valid
pointsources_test:     pointsources_testset\
                                test__normalization

#--------------------------------------------------------

pointsources_fft: ./data/pointsources_fft_done
pointsources_fft_quick: ./data/pointsources_fft_done_quick
pointsources_samp_train: ./data/pointsources_samp_train.h5
pointsources_samp_valid: ./data/pointsources_samp_valid.h5
calc_normalization: ./data/normalization_factors.csv
cnn_training: ./training_done
input_plot: ./data/plots/input_done
find_lr: ./lr_find
pointsources_testset: ./test_data/pointsources_testset
test_normalization: ./test_data/normalization_factors.csv
#########################################################
#	Tasks
#

./data/pointsources_fft_done:
		mkdir -p data/
		python create_pointsources_fft.py ./data\
		50000 784 20\
		-test False -noise False
		touch ./data/pointsources_fft_done

./data/pointsources_fft_done_quick:
		mkdir -p data/
		python create_pointsources_fft.py ./data\
		500 784 10\
		-test True -noise False
		touch ./data/pointsources_fft_done

./data/pointsources_samp_train.h5:
		python create_mnist_fft_sampled.py\
		data/pointsources_train.h5 data/pointsources_samp_train.h5\
		../simulations/layouts/vlba.txt -train True\
		-specific_mask True\
		-lon -80\
		-lat 50\
		-steps 50

./data/pointsources_samp_valid.h5:
		python create_mnist_fft_sampled.py\
		data/pointsources_valid.h5 data/pointsources_samp_valid.h5\
		../simulations/layouts/vlba.txt -train False\
		-specific_mask True\
		-lon -80\
		-lat 50\
		-steps 50

./data/normalization_factors.csv:
		python calculate_normalization.py\
		data/pointsources_samp_train.h5 \
		data/normalization_factors.csv

./training_done:
		mkdir -p models/
		python train_cnn.py ./data/pointsources_samp_train.h5\
		./data/pointsources_samp_valid.h5 ./models/unet_pointsources.model\
		UNet ./data/normalization_factors.csv\
		1000 1e-4 feature_loss -log False\
		-pretrained False ./models/unet_gauss.model \
		-inspection False
		touch ./training_done

./data/plots/input_done:
		mkdir -p data/
		mkdir -p data/plots
		python create_input_plot.py ./../resources/mnist.pkl.gz
		touch ./data/plots/input_done

./lr_find:
		mkdir -p models/
		python find_lr.py ./data/pointsources_samp_train.h5\
                ./data/pointsources_samp_valid.h5 ./models/unet_pointsources.model\
                UNet feature_loss ./data/normalization_factors.csv -log False\
                -pretrained False ./models/unet_pointsources.model\
                -save True

./test_data/pointsources_testset:
		mkdir -p test_data/
		python create_pointsources_testset.py ./test_data/pointsources_samp_test.h5\
                12500 15 \
                ../simulations/layouts/vlba.txt\
		-noise False -specific_mask True \
                -lon -80\
                -lat 50\
                -steps 50

./test_data/normalization_factors.csv:
		python test_normalization.py\
                test_data/pointsources_samp_test.h5 \
                test_data/normalization_factors.csv



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
