all: visualize_preprocessing \
	visualize_predictions \
	visualize_predictions_mnist

visualize_preprocessing: build/visualize_preprocessing_done
visualize_predictions: build/visualize_predictions_done
visualize_predictions_mnist: build/visualize_predictions_mnist
predictions: build/predictions_done
predictions_mnist: build/predictions_mnist
predictions_training: build/predictions_training

model=../models/unet_gauss.model
norm=../data/normalization_factors.csv

build/visualize_preprocessing_done:
	mkdir -p build
	python visualize_preprocessing.py
	touch visualize_preprocessing_done

build/visualize_predictions_done:
	mkdir -p build
	python visualize_prediction.py \
	UNet \
	$(model) \
	../data/gauss_samp_valid.h5 \
	$(norm) \
	build/prediction.pdf \
	-index 42

build/visualize_predictions_mnist:
	mkdir -p build
	python visualize_prediction.py \
	UNet \
        ../models/unet_mnist.model \
        ../data/gauss_samp_valid.h5 \
        $(norm) \
        build/using_mnist.pdf \
        -index 42

build/predictions_done:
	mkdir -p build
	python predictions.py \
	UNet \
	$(model) \
	../data/gauss_samp_valid.h5 \
	$(norm) \
	build/prediction.png \
	-num 20

build/predictions_mnist:
	mkdir -p build
	python predictions.py \
	UNet \
        ../models/unet_mnist.model \
        ../data/gauss_samp_valid.h5 \
        $(norm) \
        build/using_mnist.png \
        -num 20


# ------------------------------------------------------
new_shapes: new_shapes/visualize_gauss_done \
			new_shapes/visualize_circle_done \
			new_shapes/visualize_source_done

new_shapes/visualize_gauss_done:
	mkdir -p new_shapes
	python visualize_predictions.py \
	cnn \
	$(model) \
	test_data/gauss.hdf5 \
	$(norm) \
	new_shapes/gauss.pdf

new_shapes/visualize_circle_done:
	mkdir -p new_shapes
	python visualize_predictions.py \
	cnn \
	$(model) \
	test_data/circle.hdf5 \
	$(norm) \
	new_shapes/circle.pdf

new_shapes/visualize_source_done:
	mkdir -p new_shapes
	python visualize_predictions.py \
	cnn \
	$(model) \
	test_data/source.hdf5 \
	$(norm) \
	new_shapes/source.pdf

# ------------------------------------------------------

clean:
	rm -rf build/

clean_visualize_preprocessing:
	rm -rf build/x*.pdf
	rm -rf build/y*.pdf

clean_new_shapes:
	rm -rf new_shapes
