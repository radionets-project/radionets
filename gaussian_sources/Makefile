simulate_bundles: ./data/bundles_done


./data/bundles_done:
		mkdir -p data/
		python simulate_bundles.py \
		1 1024 128 data/gaussian_sources_ \
		-noise False
		# touch ./data/bundles_done

./data/fft_done:
		python create_fft_pairs.py data/ \
		./data -test True -noise False
		# touch ./data/mnist_fft_done


clean_data:
	rm -r data/