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
 
    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></span><span style="color:#000080;">This step clones the GroundingSam repository to access its codebase, which includes pre-built functionality for integrating GroundingDINO with the Segment Anything Model (SAM)
    </i></span></p>  
    <p><span style="color:white;">'</p></span>

    <p style="text-align: justify;"><span style="color:#000000;"><i>
    Install the required packages:
    </i></span></p>

.. code-block:: python

    pip install -r requirements.txt

Prepare ImageBind Checkpoint
-----------------------------

.. raw:: html
  
    <p style="text-align: justify;"><span style="color:#000000;"><i>
    You can download the pre-trained ImageBind model using <a href="https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth" target="_blank">this link</a><br>
    After downloading, put the downloaded file (imagebind_huge.pth) in [./pretrained_ckpt/imagebind_ckpt/] directory
    </i></span></p>
 
Prepare Vicuna Checkpoint
----------------------------

.. raw:: html

    <p style="text-align: justify;"><span style="color:#000000;"><i>
    To prepare the pre-trained Vicuna model, please follow the instructions provided in <a href="https://github.com/CASIA-IVA-Lab/AnomalyGPT/tree/main/pretrained_ckpt#1-prepare-vicuna-checkpoint" target="_blank">here</a><br>
    </i></span></p>


Prepare Delta Weights of AnomalyGPT
------------------------

.. raw:: html

    <p style="text-align: justify;"><span style="color:#000000;"><i>
    We use the pre-trained parameters from <a href="https://github.com/yxuansu/PandaGPT" target="_blank">PandaGPT</a> to initialize our model. You can get the weights of PandaGPT trained with different strategies in the table below.<br>
    In our experiments and online demo, we use the Vicuna-7B and <a href="https://huggingface.co/openllmplayground/pandagpt_7b_max_len_1024" target="_blank">openllmplayground/pandagpt_7b_max_len_1024</a> due to the limitation of computation resource. Better results are expected if switching to Vicuna-13B.<br>
    After that, put the downloaded 7B/13B delta weights file (pytorch_model.pt) in the ./pretrained_ckpt/pandagpt_ckpt/7b/ or ./pretrained_ckpt/pandagpt_ckpt/13b/ directory.<br>
    Then, you can download AnomalyGPT weights from those links : 
    <a href="https://huggingface.co/FantasticGNU/AnomalyGPT/blob/main/train_mvtec/pytorch_model.pt" target="_blank">Unsupervised on MVTec-AD</a>
    <a href="https://huggingface.co/FantasticGNU/AnomalyGPT/blob/main/train_visa/pytorch_model.pt" target="_blank">Unsupervised on VisA</a>
    <a href="https://huggingface.co/FantasticGNU/AnomalyGPT/blob/main/train_supervised/pytorch_model.pt" target="_blank">Supervised on MVTec-AD, VisA, MVTec-LOCO-AD and CrackForest</a>
    </i></span></p>



.. raw:: html
  
    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></span><span style="color:#000080;">Mounts Google Drive to access a shared file containing custom-trained model weights. These weights are downloaded and saved into the weights folder using the gdown library
        </i></span></p>  
    <p><span style="color:white;">'</p></span>


Install the Segment Anything Model (SAM)
-----------------------------------------------

.. code-block:: python

    !pip install 'git+https://github.com/facebookresearch/segment-anything.git'


.. raw:: html

     </i></span></p>     
    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></span><span style="color:#000080;"> 
    Installs the SAM library directly from its GitHub repository to enable segmentation functionality
        </i></span></p>  
    <p><span style="color:white;">'</p></span>



 
Download SAM Pre-trained Weights
---------------------------------------

.. code-block:: python

    %cd ./weights
    !wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    %cd {HOME}

.. raw:: html
    
    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></i></span></p>Downloads pre-trained weights for SAM from Facebook's public file repository and saves them into the weights folder
        </i></span></p> 
    <p><span style="color:white;">'</p></span>


Import and Initialize the GroundingSam Library
-------------------------------

.. code-block:: python

   from GroundingSam import *
    classes = ['crack']
    groundingsam = GroundingSam(classes=classes)


.. raw:: html
   
    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></span><span style="color:#000080;">Imports the GroundingSam library and initializes it with the class names (e.g., crack) that will be used for object detection and segmentation tasks.
   </i></span></p>
    


    <p><span style="color:white;">'</p></span>


Run Detection
-----------------------------------

.. code-block:: python

   detections = groundingsam.get_detections()




.. raw:: html
   
    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></span><span style="color:#000080;">Generates object detections using the GroundingDINO model integrated into the GroundingSam library.
      </i></span></p>
    <p><span style="color:white;">'</p></span>



Annotate Images
-----------------------------

.. code-block:: python

    groundingsam.annotate_images()

.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i>  
    - <strong>Objective</strong>: </span><span style="color:#000080;">
    Annotates the detected objects on the images. This step overlays bounding boxes or masks on the images based on the detected objects.
    <p><span style="color:white;">'</p></span>

.. figure:: /Documentation/images/annotate_cast.png
   :width:  700
   :align: center
   :alt: Alternative Text

.. raw:: html

    <p><span style="color:white;">'</p></span>

Generate Segmentation Masks
---------------------------------------

.. code-block:: python

    groundingsam.get_masks()

.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i>  
    - <strong>Objective</strong>: </span><span style="color:#000080;">
    Generates segmentation masks for the detected objects using the SAM model. These masks are used for detailed segmentation of the objects within the images.
    <p><span style="color:white;">'</p></span>
    
.. figure:: /Documentation/images/mask_cast.png
   :width:  700
   :align: center
   :alt: Alternative Text

.. raw:: html

    <p><span style="color:white;">'</p></span>
