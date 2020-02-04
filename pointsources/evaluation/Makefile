all: blob_detection\
              visualize_blobs 

average_loss: build/average_loss
blob_detection: build/blob_detection
visualize_blobs: build/visualize_blobs

model=../models/unet_pointsources.model
norm=../data/normalization_factors.csv


###########################################################
# Tasks

build/average_loss:
	mkdir -p results/
	python average_loss.py \
	UNet \
	$(model) \
	../data/pointsources_samp_valid.h5 \
	$(norm) \
	results/average_loss.csv \

build/blob_detection:
	mkdir -p results/
	python blob_detection.py \
	UNet \
	$(model) \
	../data/pointsources_samp_valid.h5 \
	$(norm) \
	results/blob_detection.csv \

build/visualize_blobs:
	mkdir -p results/
	python visualize_blobs.py \
	UNet \
	$(model) \
	../data/pointsources_samp_valid.h5 \
	$(norm) \
	results/blob_detection.png \
	-num 20


#-----------------------------------------

clean:
	rm -rf results/
