-include .makerc
all: visualize_predictions

visualize_predictions: build/visualize_predictions_done
save_predictions: ./save_pred

build/visualize_predictions_done:
	mkdir -p ${out_path}
	python visualize_predictions.py \
	${arch} \
	${path_model} \
	${data} \
	${out_path} \
	-fourier ${fourier} \
	${mode}

./save_pred:
	mkdir -p ${out_path}
	python save_predictions.py ${data} \
	${arch} ${path_model} ${out_path} -fourier ${fourier} ${mode}

clean:
	rm -rf build/

