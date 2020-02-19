-include .makerc
all: visualize_predictions

visualize_predictions: build/visualize_predictions_done
save_predictions: ./save_pred

build/visualize_predictions_done:
	mkdir -p build
	python visualize_predictions.py \
	${arch} \
	${path_model} \
	${data} \
	${norm_path} \
	${out_path} \
	${mode}

./save_pred:
	python save_predictions.py ${data} \
	${arch} ${out_path} ${mode}

clean:
	rm -rf build/

