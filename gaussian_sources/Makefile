-include .makerc
dataset: simulate_bundles\
		 fft_samp

simulate_bundles: ./data/bundles_train_done\
				  ./data/bundles_valid_done
fft_samp: ./data/fft_train_done\
		  ./data/fft_valid_done
calc_normalization: ./data/normalization_factors.csv
cnn_training: ./training_done
find_lr: ./lr_find

######################################################################

./data/bundles_train_done:
		mkdir -p data/
		python simulate_bundles.py \
		80 1024 64 data/gaussian_sources_train \
		-noise False
		touch ./data/bundles_train_done

./data/fft_train_done:
		python create_fft_pairs.py \
		./data ./data/fft_samp_ ../simulations/layouts/vlba.txt \
		-train True \
		-samp True \
		-specific_mask True \
		-lon -80 \
		-lat 50 \
		-steps 50
		touch ./data/fft_train_done

./data/bundles_valid_done:
		mkdir -p data/
		python simulate_bundles.py \
		20 1024 64 data/gaussian_sources_valid \
		-noise False
		touch ./data/bundles_valid_done

./data/fft_valid_done:
		python create_fft_pairs.py \
		./data ./data/fft_samp_ ../simulations/layouts/vlba.txt \
		-train False \
		-samp True \
		-specific_mask True \
		-lon -80 \
		-lat 50 \
		-steps 50
		touch ./data/fft_valid_done

./data/normalization_factors.csv:
		python calculate_normalization.py\
		data/ \
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

clean_norm:
	rm -r ${norm_path}

clean_data:
	rm -r data/
