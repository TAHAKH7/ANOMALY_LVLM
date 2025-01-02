Running AnomalyGPT Demo
========================================

---------------------------------------------------------------------------------------------------------------------------------


1-Environment Installation
----------------------

.. raw:: html

    <p style="text-align: justify;"><span style="color:#000000;"><i>
    Clone the repository locally:
    </i></span></p>

.. code-block:: python

    git clone https://github.com/TAHAKH7/ANOMALY_LVLM.git


.. raw:: html

    <p style="text-align: justify;"><span style="color:#000000;"><i>
    Install the required packages:
    </i></span></p>

.. code-block:: python

    pip install -r requirements.txt

2-Prepare ImageBind Checkpoint
-----------------------------

.. raw:: html
  
    <p style="text-align: justify;"><span style="color:#000000;"><i>
    You can download the pre-trained ImageBind model using <a href="https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth" target="_blank">this link</a><br>
    After downloading, put the downloaded file (imagebind_huge.pth) in [./pretrained_ckpt/imagebind_ckpt/] directory
    </i></span></p>
 
3-Prepare Vicuna Checkpoint
----------------------------

.. raw:: html

    <p style="text-align: justify;"><span style="color:#000000;"><i>
    To prepare the pre-trained Vicuna model, please follow the instructions provided in <a href="https://github.com/CASIA-IVA-Lab/AnomalyGPT/tree/main/pretrained_ckpt#1-prepare-vicuna-checkpoint" target="_blank">here</a><br>
    </i></span></p>


4-Prepare Delta Weights of AnomalyGPT
------------------------

.. raw:: html

    <p style="text-align: justify;"><span style="color:#000000;"><i>
    We use the pre-trained parameters from <a href="https://github.com/yxuansu/PandaGPT" target="_blank">PandaGPT</a> to initialize our model. You can get the weights of PandaGPT trained with different strategies in the table below.<br>
    In our experiments and online demo, we use the Vicuna-7B and <a href="https://huggingface.co/openllmplayground/pandagpt_7b_max_len_1024" target="_blank">openllmplayground/pandagpt_7b_max_len_1024</a> due to the limitation of computation resource. Better results are expected if switching to Vicuna-13B.<br>
    After that, put the downloaded 7B/13B delta weights file (pytorch_model.pt) in the ./pretrained_ckpt/pandagpt_ckpt/7b/ or ./pretrained_ckpt/pandagpt_ckpt/13b/ directory.<br>
    Then, you can download AnomalyGPT weights from those links : 
    <a href="https://huggingface.co/FantasticGNU/AnomalyGPT/blob/main/train_mvtec/pytorch_model.pt" target="_blank">Unsupervised on MVTec-AD</a><br>
    <a href="https://huggingface.co/FantasticGNU/AnomalyGPT/blob/main/train_visa/pytorch_model.pt" target="_blank">Unsupervised on VisA</a><br>
    <a href="https://huggingface.co/FantasticGNU/AnomalyGPT/blob/main/train_supervised/pytorch_model.pt" target="_blank">Supervised on MVTec-AD, VisA, MVTec-LOCO-AD and CrackForest</a><br>
    </i></span></p>



5-Deploying Demo
-----------------------------------------------
.. raw:: html

    <p style="text-align: justify;"><span style="color:#000000;"><i>
    Upon completion of previous steps, you can run the demo locally as
    </i></span></p>

.. code-block:: python

    cd ./code/
    python web_demo.py

 
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
    data
    |---pandagpt4_visual_instruction_data.json
    |---images
    |-----|-- ...
    |---mvtec_anomaly_detection
    |-----|-- bottle
    |-----|-----|----- ground_truth
    |-----|-----|----- test
    |-----|-----|----- train
    |-----|-- capsule
    |-----|-- ...
    |----VisA
    |-----|-- split_csv
    |-----|-----|--- 1cls.csv
    |-----|-----|--- ...
    |-----|-- candle
    |-----|-----|--- Data
    |-----|-----|-----|----- Images
    |-----|-----|-----|--------|------ Anomaly 
    |-----|-----|-----|--------|------ Normal 
    |-----|-----|-----|----- Masks
    |-----|-----|-----|--------|------ Anomaly 
    |-----|-----|--- image_anno.csv
    |-----|-- capsules
    |-----|-----|----- ...
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

