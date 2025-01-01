Neural Networks
=============

----------------------------------------------------------------------------------------------------------------------------------------------





.. raw:: html

    <p><span style="color:white;">'</p></span>

What is Neural Networks ?
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
    <p><span style="color:rgb(41, 128, 185);"><b>1. Vision-Language Alignment<b></span></p>

    Neural networks bridge the gap between visual and textual data by integrating features from CNNs and embedding them into the Vicuna-7B LLM for generating responses.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p style="color:red; margin-bottom: 8px;"><b>How it Works:</b></p>
    Visual features are transformed into embeddings and aligned with textual prompts using linear layers and a prompt learner module.<br>
    The neural network ensures that visual anomaly data can be seamlessly processed alongside text-based user queries.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p style="color:red; margin-bottom: 8px;"><b>Example:</b></p>
    Input: A segmentation map of an anomalous region.<br>
    Neural Network Task: --Converts the map into embeddings and aligns it with the query:-- "What anomaly is present?"<br>
    Output: "A crack in the lower-left corner."<br>
    <p><span style="color:white;">'</p></span>

    <p><span style="color:rgb(41, 128, 185);"><b>2. Prompt Embedding with Fine-Grained Semantic<b></span></p>

    The Prompt Learner is a neural network that transforms localization results into embeddings compatible with the language model.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p style="color:red; margin-bottom: 8px;"><b>How it Works:</b></p>
    The prompt learner uses a CNN to process pixel-level localization results and converts them into embeddings.<br>
    These embeddings, combined with base prompt embeddings, provide context for anomaly descriptions.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p style="color:red; margin-bottom: 8px;"><b>Example:</b></p>
    Input: A heatmap indicating an anomaly.<br>
    Output: --Prompt embeddings that the language model interprets as:-- "The anomaly is a surface dent in the upper region."<br>
    <p><span style="color:white;">'</p></span>

    <p><span style="color:rgb(41, 128, 185);"><b>3. Few-Shot and In-Context Learning<b></span></p>
    Neural networks store and retrieve patch-level features in memory banks for few-shot learning scenarios, allowing the system to adapt to new datasets with minimal data.<br>
    <p style="color:red; margin-bottom: 8px;"><b>How it Works:</b></p>
    Features from a few normal samples are stored in a memory bank.<br>
    During inference, the neural network compares these stored features with test samples to localize anomalies.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p style="color:red; margin-bottom: 8px;"><b>Example:</b></p>
    Reference: A normal screw.<br>
    Test Image: A screw with a missing section.<br>
    Neural Network Task: Detects and highlights the missing section as an anomaly.<br>
    <p><span style="color:white;">'</p></span>
    
    </i></span></p>

.. raw:: html

    <p><span style="color:white;">'</p></span>
    

