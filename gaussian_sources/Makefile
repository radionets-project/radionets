-include .makerc
dataset: simulate_bundles\
		 fft_samp

simulate_bundles: ./data3/bundles_train_done\
				  ./data3/bundles_valid_done\
				  ./data3/bundles_test_done
fft_samp: ./data3/fft_train_done\
		  ./data3/fft_valid_done\
		  ./data3/fft_test_done
calc_normalization: ./data3/normalization_factors.csv
cnn_training: ./training_done
find_lr: ./lr_find

######################################################################

./data3/bundles_train_done:
		mkdir -p data3/
		python simulate_bundles.py \
		160 512 128 data3/gaussian_sources_train \
		-noise False
		touch ./data3/bundles_train_done

./data3/fft_train_done:
		python create_fft_pairs.py \
		./data3 ./data3/fft_samp_ ../simulations/layouts/vlba.txt \
		-mode train \
		-samp True \
		-size 128 \
		-specific_mask True \
		-lon -80 \
		-lat 50 \
		-steps 50
		touch ./data3/fft_train_done

./data3/bundles_valid_done:
		mkdir -p data3/
		python simulate_bundles.py \
		40 512 128 data3/gaussian_sources_valid \
		-noise False
		touch ./data3/bundles_valid_done

./data3/fft_valid_done:
		python create_fft_pairs.py \
		./data3 ./data3/fft_samp_ ../simulations/layouts/vlba.txt \
		-mode valid \
		-samp True \
		-size 128 \
		-specific_mask True \
		-lon -80 \
		-lat 50 \
		-steps 50
		touch ./data3/fft_valid_done

./data3/bundles_test_done:
		mkdir -p data3/
		python simulate_bundles.py \
		40 512 128 data3/gaussian_sources_test \
		-noise False
		touch ./data3/bundles_test_done

./data3/fft_test_done:
		python create_fft_pairs.py \
		./data3 ./data3/fft_samp_ ../simulations/layouts/vlba.txt \
		-mode test \
		-samp True \
		-size 128 \
		-specific_mask True \
		-lon -80 \
		-lat 50 \
		-steps 50
		touch ./data3/fft_test_done

./data3/normalization_factors.csv:
		python calculate_normalization.py\
		data3/ \
		${norm_path}

./training_done:
		mkdir -p ${path}
		python train_cnn.py ${data}\
		${path_model}\
		${arch} ${norm_path}\
		${epochs} ${lr} ${lr_type} -log ${log}\
		-pretrained ${pretrained} ${pre_path}\
		-inspection ${inspection}
		# touch ./training_done

./lr_find:
		mkdir -p models/
		python find_lr.py ${data}\
		${arch} ${lr_type} ${norm_path} -min_lr ${min_lr}\
		-max_lr ${max_lr} -log ${log}\
		-pretrained ${pretrained} ${pre_path}\
		-save ${save}

./plot_loss:
		mkdir -p ${path}
		python plot_loss.py\
		${path_model}\
		${arch}\

clean_norm:
	rm -r ${norm_path}

clean_data:
	rm -r data/
