all: visualize_preprocessing

visualize_preprocessing: build/visualize_preprocessing_done

build/visualize_preprocessing_done:
	mkdir -p build
	python visualize_preprocessing.py

clean:
	rm -rf build/

clean_visualize_preprocessing:
	rm -rf build/x*.pdf
	rm -rf build/y*.pdf