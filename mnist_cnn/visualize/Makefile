all: visualize_preprocessing \
	visualize_predictions

visualize_preprocessing: build/visualize_preprocessing_done
visualize_predictions: build/visualize_predictions_done

build/visualize_preprocessing_done:
	mkdir -p build
	python visualize_preprocessing.py
	touch visualize_preprocessing_done

build/visualize_preprocessing_done:
	mkdir -p build
	python visualize_predections.py

clean:
	rm -rf build/

clean_visualize_preprocessing:
	rm -rf build/x*.pdf
	rm -rf build/y*.pdf