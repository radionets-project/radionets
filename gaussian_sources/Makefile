dataset: simulate_bundles\
		 fft_samp

simulate_bundles: ./data/bundles_done
fft_samp: ./data/fft_done
calc_normalization: ./data/normalization_factors.csv


./data/bundles_done:
		mkdir -p data/
		python simulate_bundles.py \
		20 1024 128 data/gaussian_sources_ \
		-noise False
		touch ./data/bundles_done

./data/fft_done:
		python create_fft_pairs.py \
		./data ./data/fft_samp_ ../simulations/layouts/vlba.txt \
		-samp True \
		-specific_mask True \
		-lon -80 \
		-lat 50 \
		-steps 50
		touch ./data/fft_done

./data/normalization_factors.csv:
		python calculate_normalization.py\
		data/ \
		data/normalization_factors.csv

clean_data:
	rm -r data/
