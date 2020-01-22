simulate_bundles: ./data/bundles_done


./data/bundles_done:
		mkdir -p data/
		python simulate_bundles.py \
		2 1024 128 data/gaussian_sources_ \
		-noise False
		# touch ./data/bundles_done


clean_data:
	rm -r data/