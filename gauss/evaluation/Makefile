all: blob_detection\
              ssim\
              msssim\
              average_ssim\
              average_msssim\
              visualize_blobs 

average_loss: build/average_loss
blob_detection: build/blob_detection
visualize_blobs: build/visualize_blobs
ssim: build/ssim
msssim: build/msssim
average_ssim: build/average_ssim
average_msssim: build/average_msssim

model=../models/unet_gauss.model
norm=../data/normalization_factors.csv


###########################################################
# Tasks

build/average_loss:
	mkdir -p results/
	python average_loss.py \
	UNet \
	$(model) \
	../data/gauss_samp_valid.h5 \
	$(norm) \
	results/average_loss.csv \

build/blob_detection:
	mkdir -p results/
	python blob_detection.py \
	UNet \
	$(model) \
	../data/gauss_samp_valid.h5 \
	$(norm) \
	results/blob_detection.csv \

build/visualize_blobs:
	mkdir -p results/
	python visualize_blobs.py \
	UNet \
	$(model) \
	../data/gauss_samp_valid.h5 \
	$(norm) \
	results/blob_detection.png \
	-num 20

build/ssim:
	mkdir -p results/
	python ssim.py \
	UNet \
	$(model) \
	../data/gauss_samp_valid.h5 \
	$(norm) \
	results/ssim.csv \
	-threshold 0.95


build/msssim:
	mkdir -p results/
	python msssim.py \
	UNet \
	$(model) \
	../data/gauss_samp_valid.h5 \
	$(norm) \
	results/msssim.csv \
	-threshold 0.975

build/average_ssim:
	mkdir -p results/
	python average_ssim.py \
	UNet \
	$(model) \
	../data/gauss_samp_valid.h5 \
	$(norm) \
	results/average_ssim.csv \
	
build/average_msssim:
	mkdir -p results/
	python average_msssim.py \
	UNet \
	$(model) \
	../data/gauss_samp_valid.h5 \
	$(norm) \
	results/average_msssim.csv \
	

#-----------------------------------------

clean:
	rm -rf results/
