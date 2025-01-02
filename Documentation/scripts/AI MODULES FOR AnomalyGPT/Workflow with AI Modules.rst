Workflow with AI Modules
=============

----------------------------------------------------------------------------------------------------------------------------------------------



.. raw:: html


    <p style="text-align: justify;"><span style="color:#000000;"><i>
    <p><span style="color:rgb(41, 128, 185);"><b>Input<b></span></p>
    Visual Input: An industrial image, such as a screw.<br>
    Textual Query: A user asks, "Is there any anomaly in the image?"<br>
    <p><span style="color:white;">'</p></span>
    <p style="margin: 8px;"><span style="color:white;"></span></p>
    <p style="color:red; margin-bottom: 8px;"><b>Step 1: Image Processing (Computer Vision & CNNs)</b></p>
    <p style="color:green; margin-bottom: 8px;"><b>Feature Extraction with ImageBind:</b></p>
    The ImageBind-Huge model processes the input image.<br>
    Hierarchical features are extracted, capturing both global and local details like texture, shape, and structure of the screw.<br>
    These features are converted into intermediate embeddings (Fimg) for further analysis.<br>
    <p><span style="color:white;">'</p></span>
    <p style="color:green; margin-bottom: 8px;"><b>Anomaly Localization with the Decoder:</b></p>
    The Feature-Matching Decoder compares the extracted features (Fimg) against reference features stored in memory banks (e.g., features of a normal screw).<br>
    Deviations between the query image and the normal reference image are identified.<br>
    A pixel-level localization map is generated, highlighting regions with anomalies (e.g., missing threads).<br>
    <p><span style="color:white;">'</p></span>
    <p><span style="color:rgb(41, 128, 185);"><b>Output<b></span></p>
    Visual: A localization map (heatmap) showing detected anomalies.
    Data: Feature embeddings that will be processed in subsequent steps.
    <p><span style="color:white;">'</p></span>
    <p style="color:red; margin-bottom: 8px;"><b>Step 2: Feature Alignment (Neural Networks)</b></p>
    <p style="color:green; margin-bottom: 8px;"><b>Embedding Conversion:</b></p>
    The Prompt Learner, a neural network module, processes the localization map and extracted features.<br>
    It converts these into prompt embeddings (Eprompt), which are aligned with the text input format required by the Vicuna-7B language model.<br>
    <p><span style="color:white;">'</p></span>
    <p style="color:green; margin-bottom: 8px;"><b>Alignment with Textual Context:</b></p>
    The neural network ensures that visual features are compatible with the textual context of the user query.<br>
    This alignment bridges the gap between the image analysis (visual data) and query understanding (text data).<br>
    <p><span style="color:white;">'</p></span>
    <p><span style="color:rgb(41, 128, 185);"><b>Output<b></span></p>
    A structured embedding (Eprompt) that encapsulates both the anomaly localization results and the visual context.<br>
    <p><span style="color:white;">'</p></span>
    <p style="color:red; margin-bottom: 8px;"><b>Step 3: Query Handling (NLP)</b></p>
    <p style="color:green; margin-bottom: 8px;"><b>Interpreting the Query:</b></p>
    The Vicuna-7B model synthesizes the visual and textual data.<br>
    <p><span style="color:white;">'</p></span>
    <p style="color:green; margin-bottom: 8px;"><b>Generating the Response:</b></p>
    Based on the anomaly localization results, it generates a natural language response that describes the detected anomaly.<br>
    <p><span style="color:white;">'</p></span>
    <p><span style="color:rgb(41, 128, 185);"><b>Output<b></span></p>
    Textual Response: "Yes, the screw is missing threads near the middle."<br>
    <p><span style="color:white;">'</p></span>
    <p style="color:red; margin-bottom: 8px;"><b>Step 2: Feature Alignment (Neural Networks)</b></p>
    <p style="color:green; margin-bottom: 8px;"><b>User Follow-Up Query:</b></p>
    The user asks, "Can you highlight the location?"<br>
    This query is passed back into the system, triggering a new processing cycle to refine the output.<br>
    <p><span style="color:white;">'</p></span>
    <p style="color:green; margin-bottom: 8px;"><b>Generating Updated Output:</b></p>
    The previously generated localization map (heatmap) is updated to provide clearer visual feedback.<br>
    The map is overlaid on the original image to visually highlight the exact location of the missing threads.<br>
    <p><span style="color:white;">'</p></span>
    
    </i></span></p>

.. raw:: html

    <p><span style="color:white;">'</p></span>
    

