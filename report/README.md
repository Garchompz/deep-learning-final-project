# Report Summary
The final project report is titled **"Image Captioning Using EfficientNetV2B0 + LSTM on the Flickr 8k Dataset."**
### 1\. Introduction
  * **Background:** The project aims to address the challenge of image interpretation for visually impaired individuals and assist search engines in indexing visual content, as manual caption generation is considered tedious.
  * **Objectives:** To develop an accurate and coherent automatic *image captioning* model, analyze the role of EfficientNetV2B0 and LSTM with Bahdanau Attention, and evaluate the model's performance using standard metrics.
  * **Scope:** Building a *deep learning-based image captioning* system with an *encoder-decoder* architecture, utilizing EfficientNetV2B0 as the *visual feature extractor* and LSTM with Bahdanau Attention as the *sequence decoder*.
### 2\. Dataset
  * **Data:** The **Flickr 8k Dataset** is used, consisting of 8,091 unique images, each annotated with 5 captions, totaling 40,455 captions.
  * **Data Analysis (EDA):** The dataset was found to have consistent annotations, the vocabulary distribution follows a *long-tail pattern*, and the average caption length is 11–12 words.
### 3\. Modeling
  * **Model Architecture:** An **Encoder–Decoder** architecture is used, composed of:
      * **Encoder (CNN):** EfficientNetV2B0 to extract high-level visual features.
      * **Decoder (RNN):** LSTM (*Long Short-Term Memory*) to generate the word sequence.
      * **Attention Mechanism:** **Bahdanau Attention** is added to allow the decoder to focus on relevant image regions during word generation.
  * **Training Scenario:**
      * **Transfer Learning** is applied by freezing the weights of EfficientNetV2B0, which was pre-trained on ImageNet.
      * Training configuration includes a Batch size of 128, Adam Optimizer, Sparse Categorical Crossentropy Loss function, and regularization techniques such as Dropout, Early Stopping, and Reduce Learning Rate on Plateau.
### 4\. Evaluation
  * **Metric Results:** The most optimal model (**v3** / **v10**) showed the following results:
      * **BLEU-1:** 0.5951
      * **BLEU-4:** 0.1995
      * **METEOR:** 0.3996
      * **ROUGE-L:** 0.4517
      * **CIDEr:** 0.3013
  * **Evaluation Conclusion:** The results indicate that the model has good and stable performance. A high BLEU-1 value suggests good recognition of important words, and the METEOR/ROUGE-L/CIDEr scores show good semantic relevance and sentence structure alignment with human-generated captions.
### 5\. Deployment
  * The application is implemented with a *frontend* using Vite/React Js (deployed on Vercel) and a *backend* using Python/Flask/Docker (deployed on Hugging Face Space) to separate the computational load and optimize the UI.
### 6\. Reflection
  * **Critical Analysis & Constraints:** Although the chosen architecture is appropriate, model accuracy has room for improvement. Key constraints include the **limited scale and scope of the Flickr8k dataset**, which restricts the model's generalization capability, and the **high computational requirements** that limit the use of higher image resolutions or more complex architectures. The main takeaway is the importance of balancing data quality, architecture selection, and computational resources.

## Presentation slides link
[Presentation slides](https://www.canva.com/design/DAG7asfl8-U/QO0siIVgKTzP0cqxBzcaug/edit?utm_content=DAG7asfl8-U&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)