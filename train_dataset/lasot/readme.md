# Preprocessing LaSOT
Large-scale Single Object Tracking

### Prepare dataset

After download the dataset, please unzip the dataset at *train_dataset/lasot* directory
````shell
mkdir data
unzip LaSOT/zip/*.zip -d ./data
````

### Crop & Generate data info

````shell
#python par_crop.py [crop_size] [num_threads]
python par_crop.py 511 12
python gen_json.py
````
