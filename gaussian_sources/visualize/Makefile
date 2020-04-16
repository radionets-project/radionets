-include .makerc
all: save_predictions visualize_predictions

visualize_predictions: build/visualize_predictions_done
save_predictions: ./pred_save

build/visualize_predictions_done:
	mkdir -p ${out_path}
	mkdir -p ${out_path}diff/
	python visualize_predictions.py \
	${out_path} \
	-fourier ${fourier} -amp_phase ${amp_phase} -diff ${diff} \
	${mode}

./pred_save:
	mkdir -p ${out_path}
	python save_predictions.py ${data} \
	${arch} ${path_model} ${out_path} -fourier ${fourier} ${mode}

clean:
	rm -rf build/

