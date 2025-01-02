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

 
