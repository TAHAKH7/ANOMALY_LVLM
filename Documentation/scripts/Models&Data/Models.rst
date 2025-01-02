Pre-Trained Models
============================




1. Pre-Trained ImageBind Model
-------------

.. raw:: html

    <p style="text-align: justify;"><span style="color:#000000;">
    The ImageBind-Huge model is used as the image encoder in the AnomalyGPT architecture. This pre-trained model is designed to align features across multiple modalities, including images, text, and audio. In AnomalyGPT, it processes industrial images to extract high-level and patch-level features necessary for detecting anomalies.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p style="color:red; margin-bottom: 8px;"><b>Contribution to the Project:</b></p>
    <p style="text-align: justify;"><span style="color:#000000;">
    Extracts hierarchical visual features from input images, which are later used for anomaly localization and embedding generation.<br>
    Outputs features from intermediate layers for patch-level comparison and anomaly localization using a decoder.<br>
    Ensures multi-modal compatibility by aligning image features with textual descriptions, enabling seamless integration with the large language model (LLM).<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    
    </span></p>



2. Pre-Trained Vicuna Model
-------------

.. raw:: html

    <p style="text-align: justify;"><span style="color:#00008B;">
    The Vicuna-7B model is a pre-trained large language model (LLM) utilized for generating natural language responses based on user queries and visual embeddings. It is designed for interactive, multi-turn dialogue capabilities.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p style="color:red; margin-bottom: 8px;"><b>Contribution to the Project:</b></p>
    <p style="text-align: justify;"><span style="color:#000000;">
    Processes prompt embeddings, image embeddings, and user inputs to generate human-like textual responses.<br>
    Enables the system to interpret and respond to user queries about detected anomalies, such as their presence, location, and severity.<br>
    Facilitates the integration of visual and textual data by leveraging fine-tuned prompt embeddings to improve anomaly detection precision.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    
    </span></p>



3. Pre-Trained Parameters from PandaGPT
------------------------------


.. raw:: html

    <p style="text-align: justify;"><span style="color:#000000;">
    The pre-trained parameters from PandaGPT are used to initialize the AnomalyGPT model. PandaGPT connects ImageBind with Vicuna and supports multi-modal inputs.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p style="color:red; margin-bottom: 8px;"><b>Contribution to the Project:</b></p>
    <p style="text-align: justify;"><span style="color:#000000;">
    Provides a strong starting point for multi-modal understanding by leveraging pre-trained weights that have already been fine-tuned on general visual-textual alignment tasks.<br>
    Preserves transferability and prevents catastrophic forgetting during the fine-tuning process with industrial anomaly detection (IAD) data.<br>
    Facilitates the alignment of visual and textual modalities for accurate anomaly descriptions and dialogue responses.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>

    
    </span></p>

