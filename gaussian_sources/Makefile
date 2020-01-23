dataset: simulate_bundles\
		 fft_samp

simulate_bundles: ./data/bundles_done
fft_samp: ./data/fft_done


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


clean_data:
	rm -r data/
