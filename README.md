\# FL-DTACE: A Hierarchical Federated Learning Pipeline for Low-Latency IoT Edge Security



This repository contains the official implementation of the FL-DTACE security pipeline for IoT edge networks. It includes a federated TinyML classifier, a Dynamic Trust and Anomaly Control Engine (DTACE), and an edge-network latency simulation environment.



\*\*Dataset\*\*

We used the DNN version of the Edge-IIoTset dataset (2.2 million packets). Due to its large size, it is not included in this repository. You can download the dataset directly from Kaggle here:

\[Edge-IIoTset Cyber Security Dataset](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot)



\*\*How to Run\*\*

1\. Install the required dependencies:

&#x20;  pip install -r requirements.txt

2\. Download the dataset from Kaggle and update the `CSV\_PATH` inside the configuration cell.

3\. Run the Python script or the Jupyter Notebook:

&#x20;  python fl\_dtace\_v3.py

