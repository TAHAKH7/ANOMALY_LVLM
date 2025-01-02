Evaluation
============================





1. Evaluation Metrics
-------------

.. raw:: html

    <p style="text-align: justify;"><span style="color:#000000;">
    <p style="color:blue; margin-bottom: 8px;"><b>Image-Level AUC (Area Under the ROC Curve):</b></p>
    Measures the model's ability to classify an entire image as normal or anomalous.<br>
    A higher AUC indicates better binary classification performance.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p style="color:red; margin-bottom: 8px;"><b>Results:</b></p>
    MVTec-AD: 97.4%
    VisA (Few-Shot): 87.4%

    <p style="color:blue; margin-bottom: 8px;"><b>Pixel-Level AUC:</b></p>
    Evaluates the precision of anomaly localization by comparing the predicted heatmap to ground truth anomaly masks.<br>
    Higher values reflect better localization accuracy.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p style="color:red; margin-bottom: 8px;"><b>Results:</b></p>
    MVTec-AD: 93.1%
    VisA (Few-Shot): 96.2%

    <p style="color:blue; margin-bottom: 8px;"><b>Accuracy:</b></p>
    The percentage of correctly classified samples (normal or anomalous) during testing.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p style="color:red; margin-bottom: 8px;"><b>Results:</b></p>
    MVTec-AD: 93.3%<br>
    VisA (Few-Shot): 77.4%<br>

    
    </span></p>



2. Ablation Studies
-------------

.. raw:: html

    <p style="text-align: justify;"><span style="color:#00008B;">
    <p style="color:blue; margin-bottom: 8px;"><b>Purpose:</b></p>
    Ablation studies test the contributions of individual components to the overall performance of AnomalyGPT. This includes the Feature-Matching Decoder, Prompt Learner, and Few-Shot Learning modules.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p style="color:blue; margin-bottom: 8px;"><b>Findings:</b></p>
    <p style="color:red; margin-bottom: 8px;"><b>Feature-Matching Decoder:</b></p>
    Contributed significantly to pixel-level localization accuracy, enabling precise anomaly mapping.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p style="color:red; margin-bottom: 8px;"><b>Prompt Learner:</b></p>
    Improved alignment between visual embeddings and textual prompts, enhancing response quality and localization precision.<br>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p style="color:red; margin-bottom: 8px;"><b>Few-Shot Learning:</b></p>
    Demonstrated robust performance with minimal training data, achieving high pixel-level AUC on VisA dataset anomalies.<br>


    
    </span></p>


