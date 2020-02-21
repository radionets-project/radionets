all: blob_detection\
              average_ssim\
              average_msssim\
              iqr_ssim\
              iqr_msssim\
              visualize_blobs\
              peak_detection\
              visualize_peaks 

average_loss: results/average_loss
blob_detection: results/blob_detection
visualize_blobs: results/visualize_blobs
ssim: results/ssim
msssim: results/msssim
average_ssim: results/average_ssim
average_msssim: results/average_msssim
iqr_ssim: results/iqr_ssim
iqr_msssim: results/iqr_msssim
peak_detection: results/peak_detection
visualize_peaks: results/visualize_peaks

model=../models/unet_pointsources.model
data=../test_data/pointsources_samp_test.h5
norm=../test_data/normalization_factors.csv


###########################################################
# Tasks

results/average_loss:
	mkdir -p results/
	python average_loss.py \
	UNet \
	$(model) \
	$(data) \
	$(norm) \
	results/average_loss.csv \

results/blob_detection:
	mkdir -p results/
	python blob_detection.py \
	UNet \
	$(model) \
	$(data) \
	$(norm) \
	results/blob_detection.csv \

results/visualize_blobs:
	mkdir -p results/
	python visualize_blobs.py \
	UNet \
	$(model) \
	$(data) \
	$(norm) \
	results/blob_detection.png \
	-num 20

results/ssim:
	mkdir -p results/
	python ssim.py \
	UNet \
	$(model) \
	$(data) \
	$(norm) \
	results/ssim.csv \
	-threshold 0.95


results/msssim:
	mkdir -p results/
	python msssim.py \
	UNet \
	$(model) \
	$(data) \
	$(norm) \
	results/msssim.csv \
	-threshold 0.975

results/average_ssim:
	mkdir -p results/
	python average_ssim.py \
	UNet \
	$(model) \
	$(data) \
	$(norm) \
	results/average_ssim.csv \
	
results/average_msssim:
	mkdir -p results/
	python average_msssim.py \
	UNet \
	$(model) \
	$(data) \
	$(norm) \
	results/average_msssim.csv \
	
results/iqr_ssim:
	mkdir -p results/
	python iqr_ssim.py \
        UNet \
        $(model) \
        $(data) \
        $(norm) \
        results/iqr_ssim.csv \
        
results/iqr_msssim:
	mkdir -p results/
	python iqr_msssim.py \
        UNet \
        $(model) \
        $(data) \
        $(norm) \
        results/iqr_msssim.csv \

results/peak_detection:
	mkdir -p results/
	python peak_detection.py \
	UNet \
	$(model) \
	$(data) \
	$(norm) \
	results/peak_detection.csv \

results/visualize_peaks:
	mkdir -p results/
	python visualize_peaks.py \
	UNet \
	$(model) \
	$(data) \
	$(norm) \
	results/peak_detection.png \
	-num 20



#-----------------------------------------

clean:
	rm -rf results/
