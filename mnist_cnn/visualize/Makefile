all: visualize_preprocessing \
	visualize_predictions

visualize_preprocessing: build/visualize_preprocessing_done
visualize_predictions: build/visualize_predictions_done

build/visualize_preprocessing_done:
	mkdir -p build
	python visualize_preprocessing.py
	touch visualize_preprocessing_done

build/visualize_predictions_done:
	mkdir -p build
	python visualize_predictions.py \
	cnn \
	../models/test.model \
	../data/mnist_samp_valid.h5 \
	../data/normalization_factors.csv \
	build/prediction1.pdf \
	-index 42

new_shapes: new_shapes/visualize_gauss_done \
			new_shapes/visualize_circle_done \
			new_shapes/visualize_source_done

new_shapes/visualize_gauss_done:
	mkdir -p new_shapes
	python visualize_predictions.py \
	cnn \
	../models/hamburg2.model \
	test_data/gauss.hdf5 \
	../data/normalization_factors.csv \
	new_shapes/gauss.pdf

new_shapes/visualize_circle_done:
	mkdir -p new_shapes
	python visualize_predictions.py \
	cnn \
	../models/hamburg2.model \
	test_data/circle.hdf5 \
	../data/normalization_factors.csv \
	new_shapes/circle.pdf

new_shapes/visualize_source_done:
	mkdir -p new_shapes
	python visualize_predictions.py \
	cnn \
	../models/hamburg2.model \
	test_data/source.hdf5 \
	../data/normalization_factors.csv \
	new_shapes/source.pdf

clean:
	rm -rf build/

clean_visualize_preprocessing:
	rm -rf build/x*.pdf
	rm -rf build/y*.pdf

clean_new_shapes:
	rm -rf new_shapes