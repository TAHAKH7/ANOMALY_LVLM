Neural Networks
=============

----------------------------------------------------------------------------------------------------------------------------------------------





.. raw:: html

    <p><span style="color:white;">'</p></span>

What is Neural Networks ?
----------------------------------


.. raw:: html

    <p style="text-align: justify;"><span style="color:#000000;"><i>
    Neural Networks are a class of machine learning models inspired by the structure and function of biological neural networks in the human brain. They consist of layers of interconnected nodes (neurons) that process data by applying weights and biases to inputs and passing them through activation functions. Neural networks learn patterns from data through a process called backpropagation, where errors are minimized by adjusting weights iteratively. These models excel in tasks like classification, regression, clustering, and feature extraction, making them foundational to modern artificial intelligence and deep learning.
    </i></span></p>


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
    <p style="margin: 8px;"><span style="color:white;"></span></p>
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
    

