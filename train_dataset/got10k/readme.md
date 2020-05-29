# Preprocessing GOT-10K
A Large High-Diversity Benchmark for Generic Object Tracking in the Wild

### Prepare dataset 

After download the dataset, please unzip the dataset at *train_dataset/got10k* directory
mkdir data
unzip full_data/train_data/*.zip -d ./data
````

### Crop & Generate data info

````shell
#python par_crop.py [crop_size] [num_threads]
python par_crop.py 511 12
python gen_json.py
````
