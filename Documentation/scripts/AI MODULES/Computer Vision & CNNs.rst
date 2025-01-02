Computer Vision & CNNs
=============

----------------------------------------------------------------------------------------------------------------------------------------------





.. raw:: html

    <p><span style="color:white;">'</p></span>

What is Computer Vision & CNNs ?
----------------------------------


.. raw:: html

    <p style="text-align: justify;"><span style="color:#000000;"><i>
    Computer Vision is a branch of artificial intelligence (AI) that focuses on enabling machines to interpret, analyze, and understand visual data from the world. This involves tasks such as object detection, image classification, segmentation, anomaly detection, and more. Computer vision systems use algorithms to process and extract meaningful information from images or videos, simulating how humans perceive and analyze visual inputs. Its applications include facial recognition, autonomous vehicles, medical imaging, and industrial quality control.
    </i></span></p>
    <p style="text-align: justify;"><span style="color:#000000;"><i>
    Convolutional Neural Networks (CNNs) are a type of deep learning model specifically designed for processing structured grid data like images. CNNs excel in capturing spatial hierarchies in visual data by applying convolutional layers that detect features like edges, textures, and objects. These layers allow the network to learn from raw image pixels and extract high-level patterns that are crucial for image analysis tasks. CNNs are widely used in computer vision tasks due to their ability to handle large-scale image data with high accuracy and efficiency.
    </i></span></p>
    <p style="text-align: justify;"><span style="color:#000000;"><i>
    The relationship between Computer Vision and Convolutional Neural Networks (CNNs) is deeply intertwined, as CNNs serve as the foundational models that enable most modern computer vision tasks. Computer vision defines the objectives, such as object detection, segmentation, or anomaly detection, while CNNs provide the computational framework to achieve these goals by processing and analyzing visual data. CNNs extract hierarchical features from images, identifying patterns like edges, textures, and shapes, which are essential for understanding visual inputs. For instance, in anomaly detection, computer vision sets the task of identifying defects in industrial images, and CNNs process the images to localize anomalies with precision, often generating heatmaps or segmentation outputs. Together, computer vision provides the "what to do," and CNNs deliver the "how to do it," creating powerful systems capable of interpreting and acting on complex visual data.
    </i></span></p>
    <p><span style="color:white;">'</p></span>


----------------------------------------------------------------------------------------------------------------------------------------------


Roles in Project
-------------------------------

.. raw:: html


    <p style="text-align: justify;"><span style="color:#000000;"><i>
    In the AnomalyGPT project, Convolutional Neural Networks (CNNs) and Computer Vision work in tandem to process, analyze, and interpret visual data from industrial images. Together, they form the backbone of the anomaly detection pipeline, enabling the system to extract features, localize anomalies, and provide visual outputs that can be integrated into interactive dialogues.<br>

    <p><span style="color:rgb(41, 128, 185);"><b>1. Feature Extraction for Visual Data<b></span></p>

    CNNs are employed within the ImageBind-Huge encoder, which is part of the Computer Vision pipeline, to extract hierarchical feature representations from industrial images. These features include details such as textures, shapes, and edges, which are essential for detecting anomalies.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p style="color:red; margin-bottom: 8px;"><b>How it Works:</b></p>
    Images are passed through multiple convolutional layers, where filters detect specific patterns like straight lines, curves, or irregularities.<br>
    Higher layers in the CNN extract more abstract features, such as the structure of an industrial component.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p style="color:red; margin-bottom: 8px;"><b>Example:</b></p>
    Input: An image of a metallic panel.<br>
    Output: Feature maps that capture details of the panel’s surface texture and highlight potential scratches or dents.<br>
    <p><span style="color:white;">'</p></span>

    <p><span style="color:rgb(41, 128, 185);"><b>2. Pixel-Level Anomaly Localization<b></span></p>

    The Feature-Matching Decoder, powered by CNNs, performs pixel-level anomaly detection by comparing features extracted from query images with those of normal reference images.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p style="color:red; margin-bottom: 8px;"><b>How it Works:</b></p>
    Each image is divided into patches, and CNNs analyze these patches for deviations.<br>
    A localization map is generated, highlighting areas where the query image differs significantly from normal samples.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p style="color:red; margin-bottom: 8px;"><b>Example:</b></p>
    Input: An image of a screw with missing threads.<br>
    Output: A heatmap overlaying the screw image, indicating the missing thread locations.<br>
    <p><span style="color:white;">'</p></span>

    <p><span style="color:rgb(41, 128, 185);"><b>3. Few-Shot Learning<b></span></p>
    CNNs enable the system to adapt to new datasets with minimal normal samples by storing and comparing patch-level features in memory banks.<br>
    <p style="color:red; margin-bottom: 8px;"><b>How it Works:</b></p>
    Features of normal reference images are stored in a memory bank.<br>
    During inference, the CNNs compare these stored features with test samples to identify anomalies.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p style="color:red; margin-bottom: 8px;"><b>Example:</b></p>
    Reference Image: A normal cable.<br>
    Query Image: A frayed cable.<br>
    Output: Detection of the frayed section, with a heatmap showing its location.<br>
    <p><span style="color:white;">'</p></span>

    <p><span style="color:rgb(41, 128, 185);"><b>4. Vision-Language Integration<b></span></p>

    CNNs and Computer Vision collaborate to align visual outputs with text-based user queries, enabling multi-modal interactions.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p style="color:red; margin-bottom: 8px;"><b>How it Works:</b></p>
    CNNs generate visual feature embeddings and localization maps.<br>
    These outputs are processed by the Prompt Learner, which transforms them into embeddings compatible with the Vicuna-7B LLM.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p style="color:red; margin-bottom: 8px;"><b>Example:</b></p>
    Input: A segmentation map showing a defective area.<br>
    User Query: "What is wrong with this component?"<br>
    Output: "The component has a dent in the upper-left corner."<br>
    <p><span style="color:white;">'</p></span>

    <p><span style="color:rgb(41, 128, 185);"><b>5. Data Augmentation and Synthetic Anomaly Simulation<b></span></p>
    CNNs process augmented datasets created through techniques like Cut-Paste and Poisson Editing, allowing the model to learn from diverse and realistic synthetic anomalies.<br>
    <p style="color:red; margin-bottom: 8px;"><b>How it Works:</b></p>
    Augmented images are passed through the CNN encoder, ensuring that the model learns to detect various anomaly types.<br>
    The robustness of the system improves, enabling it to handle real-world defects effectively.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p style="color:red; margin-bottom: 8px;"><b>Example:</b></p>
    Augmented Input: A metallic component with a simulated scratch added via Poisson Editing.<br>
    Output: Accurate detection and localization of the synthetic scratch during inference.<br>
    <p><span style="color:white;">'</p></span>

    <p><span style="color:rgb(41, 128, 185);"><b>6. Multi-Modal Interaction<b></span></p>
    CNNs support Computer Vision in generating intermediate outputs, such as heatmaps and segmentation maps, that feed into the dialogue system.<br>
    <p style="color:red; margin-bottom: 8px;"><b>How it Works:</b></p>
    Visual outputs are used to generate contextual embeddings for natural language responses.<br>
    This enables detailed, follow-up interactions based on visual data.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p style="color:red; margin-bottom: 8px;"><b>Example:</b></p>
    User Query: "What is wrong with this component?"<br>
    Visual Output: A heatmap showing a crack in the upper-right corner.<br>
    Textual Output: "There is a crack in the upper-right corner of the component."<br>
    <p><span style="color:white;">'</p></span>
    
    </i></span></p>

.. raw:: html

    <p><span style="color:white;">'</p></span>
    

