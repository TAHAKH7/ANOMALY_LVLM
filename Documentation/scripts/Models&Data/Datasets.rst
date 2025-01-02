Datasets
============================



1. MVTec-AD Dataset
-------------

.. raw:: html

    <p style="color:red; margin-bottom: 8px;"><b>Description:</b></p>
    <p style="text-align: justify;"><span style="color:#000000;">
    A widely-used dataset for industrial anomaly detection consisting of 3,629 training images and 1,725 testing images across 15 categories, such as "bottle," "screw," and "wood."<br>
    Training images contain only normal samples, while testing images include both normal and anomalous samples.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p style="color:red; margin-bottom: 8px;"><b>Role in the Project:</b></p>
    Used for training and evaluating the anomaly detection system.<br>
    Facilitates unsupervised training, where only normal samples are used to learn feature distributions.<br>
    Supports one-shot transfer learning by providing a small number of normal samples for inference on unseen categories.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    </span></p>



2. VisA Dataset
-------------

.. raw:: html
    <p style="color:red; margin-bottom: 8px;"><b>Description:</b></p>
    <p style="text-align: justify;"><span style="color:#000000;">
    A newer industrial anomaly detection dataset with 9,621 normal images and 1,200 anomalous images across 12 categories, such as "candle," "capsule," and "PCB."<br>
    Offers higher resolution images (~1500×1000 pixels) than MVTec-AD.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p style="color:red; margin-bottom: 8px;"><b>Role in the Project:</b></p>
    Used for one-shot transfer experiments to test the generalization capability of the model.<br>
    Helps validate the system’s ability to adapt to different types of anomalies with minimal data.<br>    
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    </span></p>



3. Simulated Anomalous Data
------------------------------



.. raw:: html

    <p style="color:red; margin-bottom: 8px;"><b>Description:</b></p>
    <p style="text-align: justify;"><span style="color:#000000;">
    Generated using techniques like Cut-Paste and Poisson Editing to create synthetic anomalies by cropping and pasting parts of an image into different areas.<br>
    Provides a diverse range of anomalies, such as scratches, dents, or missing parts, that mimic real-world industrial defects.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p style="color:red; margin-bottom: 8px;"><b>Role in the Project:</b></p>
    Enhances the robustness of the model by increasing the variability and complexity of training data.<br>
    Improves anomaly localization and feature extraction capabilities by training the model on a broader range of defect patterns.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>

    
    </span></p>

