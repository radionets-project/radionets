-include .makerc
all: visualize_predictions

visualize_predictions: build/visualize_predictions_done

build/visualize_predictions_done:
	mkdir -p build
	python visualize_predictions.py \
	${arch} \
	${path_model} \
	${data} \
	${norm_path} \
	${out_path} \
	-fourier ${fourier} \
	${mode}

clean:
	rm -rf build/

