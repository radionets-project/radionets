dataset: simulate_bundles\
		 fft_samp

simulate_bundles: ./data2/bundles_train_done\
				  ./data2/bundles_valid_done
fft_samp: ./data2/fft_train_done\
		  ./data2/fft_valid_done
calc_normalization: ./data2/normalization_factors.csv
cnn_training: ./training_done

#############################################

norm=data/normalization_factors.csv

#############################################


./data2/bundles_train_done:
		mkdir -p data2/
		python simulate_bundles.py \
		80 1024 64 data2/gaussian_sources_train \
		-noise False
		touch ./data2/bundles_train_done

./data2/fft_train_done:
		python create_fft_pairs.py \
		./data2 ./data2/fft_samp_ ../simulations/layouts/vlba.txt \
		-train True \
		-samp True \
		-specific_mask True \
		-lon -80 \
		-lat 50 \
		-steps 50
		touch ./data2/fft_train_done

./data2/bundles_valid_done:
		mkdir -p data2/
		python simulate_bundles.py \
		20 1024 64 data2/gaussian_sources_valid \
		-noise False
		touch ./data2/bundles_valid_done

./data2/fft_valid_done:
		python create_fft_pairs.py \
		./data2 ./data2/fft_samp_ ../simulations/layouts/vlba.txt \
		-train False \
		-samp True \
		-specific_mask True \
		-lon -80 \
		-lat 50 \
		-steps 50
		touch ./data2/fft_valid_done

./data2/normalization_factors.csv:
		python calculate_normalization.py\
		data/ \
		$(norm)

./training_done:
		mkdir -p models/unet_denoise
		python train_cnn.py ./data/ \
		./models/unet_denoise/unet_denoise100.model \
		UNet_denoise $(norm) \
		100 1e-3 mse \
		-pretrained False \
		-inspection True
		# touch ./training_done

./log_done:
		mkdir -p models/unet_log
		python train_cnn.py ./data2/ \
		./models/unet_log/unet_fft_log100.model \
		UNet_fft data2/normalization_factors.csv \
		100 1e-3 mse \
		-pretrained False models/unet_log/unet_fft_log.model\
		-inspection True
		mv inspection_plot.pdf ./models/unet_log/
		# touch ./training_done

clean_norm:
	rm -r $(norm)

clean_data:
	rm -r data/
