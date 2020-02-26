-include .makerc
all: save_predictions visualize_predictions

visualize_predictions: build/visualize_predictions_done
save_predictions: ./pred_save

build/visualize_predictions_done:
	mkdir -p ${out_path}
	python visualize_predictions.py \
	${arch} \
	${path_model} \
	${out_path} \
	-fourier ${fourier} \
	${mode}

./pred_save:
	mkdir -p ${out_path}
	python save_predictions.py ${data} \
	${arch} ${path_model} ${out_path} -fourier ${fourier} ${mode}

clean:
	rm -rf build/

