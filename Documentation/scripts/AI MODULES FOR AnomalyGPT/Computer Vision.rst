Computer Vision
=============

----------------------------------------------------------------------------------------------------------------------------------------------


.. figure:: /Documentation/images/computer.jpg
   :width: 700
   :align: center
   :alt: Alternative text for the image




.. raw:: html

    <p><span style="color:white;">'</p></span>

What is Computer Vision ?
----------------------------------


.. raw:: html

    <p style="text-align: justify;"><span style="color:#000000;"><i>
    Computer Vision is a field of artificial intelligence that focuses on enabling machines to interpret, analyze, and understand visual information from the world. It involves processing images, videos, and real-time visual data to extract meaningful insights and perform tasks such as object detection, recognition, tracking, and segmentation. By mimicking the human visual system, computer vision seeks to teach machines to identify patterns, distinguish objects, and make decisions based on visual inputs. This is achieved through algorithms and models that leverage statistical techniques, machine learning, and deep learning, especially convolutional neural networks (CNNs).
    </i></span></p>
    <p style="text-align: justify;"><span style="color:#000000;"><i>
    The applications of computer vision are vast and span across multiple industries. In healthcare, it is used for medical imaging and diagnostics, such as detecting tumors or analyzing X-rays. In the automotive sector, computer vision powers autonomous vehicles by recognizing road signs, pedestrians, and other vehicles. Retail industries use it for surveillance, inventory management, and customer behavior analysis, while manufacturing employs it for quality control and anomaly detection in production lines. Recent advancements in deep learning have significantly improved the accuracy and efficiency of computer vision systems, making it an indispensable tool in creating intelligent, automated solutions.
    </i></span></p>
    <p><span style="color:white;">'</p></span>


----------------------------------------------------------------------------------------------------------------------------------------------


Roles in Project
-------------------------------

.. raw:: html


    <p style="text-align: justify;"><span style="color:#000000;"><i>

    <p><span style="color:rgb(41, 128, 185);"><b>1. Feature Extraction<b></span></p>

    Computer Vision in AnomalyGPT begins with extracting visual features from input images using the ImageBind-Huge model, a pre-trained image encoder.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p><span style="color:red;"><b>How it Works:</b></span></p>
    The image encoder processes high-resolution industrial images, converting them into hierarchical feature representations (Fimg).<br>
    These features capture important details such as textures, shapes, and patterns that are critical for identifying anomalies.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p><span style="color:red;"><b>Example:</b></span></p>
    Input: An image of a screw.<br>
    Output: A feature map highlighting the physical characteristics of the screw, such as threads, length, and surface texture.<br>
    <p><span style="color:white;">'</p></span>

    <p><span style="color:rgb(41, 128, 185);"><b>2. Pixel-Level Anomaly Localization<b></span></p>

    The Feature-Matching Decoder, a key Computer Vision module, uses extracted features to identify and localize anomalies at the pixel level.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p><span style="color:red;"><b>How it Works:</b></span></p>
    The decoder compares patch-level features of the input image with a memory bank of normal reference features.<br>
    Areas in the input image that deviate significantly from the reference are marked as anomalous.<br>
    Localization maps are generated, highlighting specific regions with anomalies.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p><span style="color:red;"><b>Example:</b></span></p>
    Input: An image of a metallic panel with scratches.<br>
    Output: A heatmap overlay on the image, highlighting the scratched regions for further inspection.<br>
    <p><span style="color:white;">'</p></span>

    <p><span style="color:rgb(41, 128, 185);"><b>3. Few-Shot Learning<b></span></p>

    <p><span style="color:red;"><b>How it Works:</b></span></p>
    During inference, the system matches query images against a few stored normal samples.<br>
    Deviations from these references are identified as potential anomalies.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p><span style="color:red;"><b>Example:</b></span></p>
    Reference Image: A normal cable.<br>
    Query Image: A frayed cable.<br>
    Output: Detection of the frayed section, with a heatmap showing its location.<br>
    <p><span style="color:white;">'</p></span>

    <p><span style="color:rgb(41, 128, 185);"><b>4. Integration with Vision-Language Models<b></span></p>

    Computer Vision bridges the gap between visual data and the Natural Language Processing (NLP) module by generating intermediate outputs that can be aligned with textual prompts.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p><span style="color:red;"><b>How it Works:</b></span></p>
    Visual outputs, such as localization maps and segmentation results, are transformed into embeddings by the Prompt Learner.<br>
    These embeddings are aligned with textual inputs for meaningful responses.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p><span style="color:red;"><b>Example:</b></span></p>
    Visual Input: A segmented image showing missing threads on a screw.<br>
    NLP Response: "The anomaly is located near the middle threads of the screw."<br>
    <p><span style="color:white;">'</p></span>

    <p><span style="color:rgb(41, 128, 185);"><b>5. Data Augmentation and Synthetic Anomaly Simulation<b></span></p>

    <p><span style="color:red;"><b>How it Works:</b></span></p>
    Synthetic anomalies are generated using techniques such as Cut-Paste and Poisson Editing, which simulate real-world defects like scratches, dents, or missing components.<br>
    These augmented datasets help the model generalize better to unseen anomalies.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p><span style="color:red;"><b>Example:</b></span></p>
    Augmented Image: A simulated defect on a metallic panel (e.g., a scratch added using Poisson Editing).<br>
    Output: The system detects and localizes the synthetic anomaly during testing.<br>
    <p><span style="color:white;">'</p></span>

    <p><span style="color:rgb(41, 128, 185);"><b>6. Multi-Modal Interaction<b></span></p>

    <p><span style="color:red;"><b>How it Works:</b></span></p>
    After generating localization maps, the visual outputs are paired with text-based user queries.<br>
    For example, the system combines a heatmap with a textual explanation, making the results more actionable.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p><span style="color:red;"><b>Example:</b></span></p>
    User Query: "What is wrong with this component?"<br>
    Visual Output: A heatmap showing a crack in the upper-right corner.<br>
    Textual Output: "There is a crack in the upper-right corner of the component."<br>
    <p><span style="color:white;">'</p></span>
    
    </i></span></p>

.. raw:: html

    <p><span style="color:white;">'</p></span>
    

