-include .makerc
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
	${arch} \
	${path_model} \
	${samp_valid} \
	${norm_path} \
	${out_path} \
	${mode}


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
