# SiameseCAR demo
**1. Structure**  

    |-- siamcar  
    
        |-- core
           |-- config.py  
           |-- xcoor.py  
    
        |-- model
            |-- backbone.py
            |-- head.py 
            |-- neck.py  
            |-- model_builder.py 
            |-- loss_car.py 
            |-- loss_cls.py 
      
        |-- tracker 
            |-- base_tracker.py 
            |-- siamcar_tracker.py 
        
        |-- utils
            |-- car_utils.py
            |-- iou_loss.py
            |-- model_load.py  
     
        |-- config.yaml
        
    |-- testing_dataset 
     
        *put the test video in this file*
     
    |-- tools
        |-- snapshot *put the pretrained model in this file*  
        |-- demp.py


**2. Environment setup**  
This code has been test on Ubuntu 16.04, Python 3.6, Pytorch 0.4.1, CUDA 9.0.   

**3. Test date**  
Put the [testing videos or picture sequences](https://pan.baidu.com/s/1qGlu1lpAEpQWGJ_bCkCwMA) into *testing_dataset* directory. During the testing, mark the tracking target in the first frame with mouse.

**4. Tracker**  
Download the pretrained model and put them into *tools/snapshot* directory.  
[general_model](https://pan.baidu.com/s/12u8YzjoAxugFTtLTk3S0JA)  
[got10k_model](https://pan.baidu.com/s/19jUavaAM47ZcgckmSv9c_Q)  
[LaSOT_model](https://pan.baidu.com/s/1HfsY335PmtMHnac_Q9jXOg)  

**5. Testing demo**
> cd SiamCAR/tools
> python demo.py  \  
        --video <testing video>  
        --snapshot <specified testing model>  
        --hp_search <hp parameters>   
        
***Optional parameters:***  
- `--video:`path of the testing video or picture sequences(eg:`--video testing_dataset/people.mp4` or `--video testing_dataset/Biker`)
- `--snapshot:`path of the specified testing model. We provide three model for testing. The general model was trained in VID, YOUTUBEBB, COCO and DET. The LaSOT model was only trained in LaSOT training dataset, and the got10k model was only trained in GOT10K training dataset.
(eg:`--snapshot ./snapshot/general_model.pth` or `--snapshot ./snapshot/LaSOT_model.pth` or `--snapshot ./snapshot/got10k_model.pth`)
- `hp_search:`There are different hyper paramters for different datasets. (eg:`--hp_search OTB/LaSOT/VOT2019/GOT10k/UAV123`) 

**5. Acknowledgement**  
We would like to express our sincere thanks to Fangyi Zhang, Qiang Wang and Bo Li. The SiamCAR was developed on the basis of [pysot](https://github.com/STVIR/pysot).