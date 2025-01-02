Train Your Own AnomalyGPT
========================================

.. raw:: html
    
    <p style="text-align: justify;"><span style="color:#000000;"><i>
    Prerequisites: Before training the model, making sure the environment is properly installed and the checkpoints of ImageBind, Vicuna and PandaGPT are downloaded.
    </i></span></p>

1-Data Preparation:
-------------------------------

.. raw:: html
   
    <p style="text-align: justify;"><span style="color:#000000;"><i>
    You can download MVTec-AD dataset from <a href="https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads" target="_blank">this link</a> and VisA from <a href="https://github.com/amazon-science/spot-diff" target="_blank">this link</a> You can also download pre-training data of PandaGPT from <a href="https://huggingface.co/datasets/openllmplayground/pandagpt_visual_instruction_dataset/tree/main" target="_blank">this link</a><br>
    After downloading, put the data in the [./data] directory.<br>
    The directory of [./data] should look like:<br>
    data<br>
    |---pandagpt4_visual_instruction_data.json<br>
    |---images<br>
    |-----|-- ...<br>
    |---mvtec_anomaly_detection<br>
    |-----|-- bottle<br>
    |-----|-----|----- ground_truth<br>
    |-----|-----|----- test<br>
    |-----|-----|----- train<br>
    |-----|-- capsule<br>
    |-----|-- ...<br>
    |----VisA<br>
    |-----|-- split_csv<br>
    |-----|-----|--- 1cls.csv<br>
    |-----|-----|--- ...<br>
    |-----|-- candle<br>
    |-----|-----|--- Data<br>
    |-----|-----|-----|----- Images<br>
    |-----|-----|-----|--------|------ Anomaly<br>
    |-----|-----|-----|--------|------ Normal<br>
    |-----|-----|-----|----- Masks<br>
    |-----|-----|-----|--------|------ Anomaly<br>
    |-----|-----|--- image_anno.csv<br>
    |-----|-- capsules<br>
    |-----|-----|----- ...<br>
    </i></span></p>
    


    <p><span style="color:white;">'</p></span>


2-Training Configurations:
-----------------------------------

.. raw:: html
   
    <p style="text-align: justify;"><span style="color:#000000;"><i>
    The table below show the training hyperparameters used in our experiments. The hyperparameters are selected based on the constrain of our computational resources, i.e. 2 x RTX3090 GPUs.<br>
    <p><span style="color:white;">'</p></span>

.. figure:: /Documentation/images/config.jpg
   :width:  700
   :align: center
   :alt: Alternative Text




3-Training AnomalyGPT:
-----------------------------

.. raw:: html
   
    <p style="text-align: justify;"><span style="color:#000000;"><i>
    To train AnomalyGPT on MVTec-AD dataset, please run the following commands:<br>
    <p><span style="color:white;">'</p></span>

.. code-block:: python

    cd ./code
    bash ./scripts/train_mvtec.sh

.. raw:: html
   
    <p style="text-align: justify;"><span style="color:#000000;"><i>
    The key arguments of the training script are as follows:<br>
    --data_path: The data path for the json file pandagpt4_visual_instruction_data.json.<br>
    --image_root_path: The root path for training images of PandaGPT.<br>
    --imagebind_ckpt_path: The path of ImageBind checkpoint.<br>
    --vicuna_ckpt_path: The directory that saves the pre-trained Vicuna checkpoints.<br>
    --max_tgt_len: The maximum sequence length of training instances.<br>
    --save_path: The directory which saves the trained delta weights. This directory will be automatically created.<br>
    --log_path: The directory which saves the log. This directory will be automatically created.<br>
    Note that the epoch number can be set in the epochs argument at ./code/config/openllama_peft.yaml file and the learning rate can be set in ./code/dsconfig/openllama_peft_stage_1.json<br>
    <p><span style="color:white;">'</p></span>

