Project Introduction
====================
------------------------------------------

AnomalyGPT: Detecting Industrial Anomalies using Large Vision-Language Models
___________________________


.. raw:: html

    <p style="text-align: justify;"><span style="color:#000080;">

    AnomalyGPT is an innovative conversational vision-language model (LVLM) designed to address Industrial Anomaly Detection (IAD). Leveraging state-of-the-art LVLMs, AnomalyGPT overcomes challenges faced by traditional IAD methods, such as reliance on manual thresholds and limited adaptability to unseen object categories. This model integrates pre-trained language and vision modules to enable automated detection, precise anomaly localization, and interactive dialogue capabilities.

   </span></p>

.. figure:: /Documentation/images/compare.jpg
   :width: 700
   :align: center
   :alt: Alternative text for the image

.. raw:: html

    <p style="text-align: justify;"><i>
    - <span style="color:blue;">Key Features :</span><span style="color:#000080;"><br>
      Automated Anomaly Judgments: Eliminates the need for manual threshold adjustments.<br>
      Pixel-Level Localization: Detects anomalies with high precision.<br>
      Few-Shot Learning: Adapts to new datasets using a single normal sample.<br>
      Multi-Turn Dialogue: Provides interactive insights for industrial anomaly detection.
    </span></i>
   </p>



    <p style="text-align: justify;"><i>

    - <span style="color:blue;">Output : </span><span style="color:#000080;">The requested object, filtred and highlighted (segmented)
    </i></span></p>
    <p style="text-align: justify;">
    <span style="color:blue;"><strong>  For example: </span></strong>
    <span style="color:#000080;"><i>
    The user has an image of an industrial product(cast) and wants to segment the crack.
    </i></span></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>
    In order to do so, the user inserts the image and writes this query: "crack"

    </i></span></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i> 

    The output would be a processed images with the crack highlighted
    </i></span></p>




.. figure:: /Documentation/images/IM.jpg
   :width: 700
   :align: center
   :alt: Alternative text for the image
   

.. raw:: html

    <p style="text-align: justify;">

    </p>

    <span style="color:blue;"><strong> How were we able to do that ?</strong></span>


    <p style="text-align: justify;"><span style="color:#000080;"><i>

    Building from scratch a model, that is trained on a dataset according to the field of interest.
    </i></span></p>

    <span style="color:blue;"><strong> What's new about the project ?</strong></span>

    <p style="text-align: justify;"><span style="color:#000080;"><i>

    Preparing an image dataset for training a model on segmentation is a time and energy consuming task, this process is done manually where one has to draw a contour on each object and label it.
    </i></span></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>

    The bridge, the connection or the model we are building from scratch uses FOUNDATION MODELS for training (look at like a human sitting on a computer, drawing contours and labeling each object on the image). This enable optimization of time and labor resources and open doors to the use of large-scale datasets for training and application purposes using flexible prompt.

    </i></span></p>


    <p style="text-align: justify;"><span style="color:#000080;"><i>
    
    This project goes way beyond the scope of detecting dogs in parks and may be used to perform object detection on any image in any field.

    </i></span></p>



    <span style="color:blue;"><strong>Project building strategy: </strong></span>
    <p style="text-align: justify;"><span style="color:#000080;"><i>
    Modular components
    </i></span></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>
    Manual implementation: Each component is implemented manually for pedagogical reasons
    </i></span></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>
    Build to last strategy : Simple, accessible documentation with practice examples
    </i></span></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>
    Accuracy-oriented: Replacing manually implemented components with imported frameworks for more accuracy

    </i></span></p>


.. raw:: html

    <p style="text-align: justify;">

    </p>


Documentation axes
_________________________

.. figure:: /Documentation/images/scope/3.jpg
   :width: 700
   :align: center
   :alt: Alternative text for the image

.. figure:: /Documentation/images/scope/4.jpg
   :width: 700
   :align: center
   :alt: Alternative text for the image
