dataset: simulate_bundles\
		 fft_samp

simulate_bundles: ./data/bundles_train_done\
				  ./data/bundles_valid_done
fft_samp: ./data/fft_train_done\
		  ./data/fft_valid_done
calc_normalization: ./data/normalization_factors.csv
cnn_training: ./training_done

#############################################

norm=data/normalization_factors.csv

#############################################


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
		$(norm)

./training_done:
		python train_cnn.py ./data/\
		./models/test_gs_400.model\
		cnn $(norm)\
		200 1e-3 mse -log False\
		-pretrained True ./models/test_gs.model\
		-inspection True
		# touch ./training_done

clean_norm:
	rm -r $(norm)

clean_data:
	rm -r data/
