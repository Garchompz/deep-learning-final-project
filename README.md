# Deep Learning Final Project
This final project is part of the Deep Learning (COMP6826001) streaming course during the 2025/2026 odd semester period in BINUS University.

Created by Group 9 Deep Learning class LE01:
1. Kent Amadeo Timotheus - 2702227025
2. Theodore Zachary - 2702244100
3. Renaldo - 2702235670

This project covers image captioning using EfficientNetV2B0 + Long-Short Term Memory (LSTM) on the Flickr 8k (and 30k) dataset. This project includes a runnable application which incorporates the model, allowing caption generation through a web app. This repository also covers the report, outputs of the model, dataset information, app code, and experiment notebooks to generate the model. 

## Deployed application
A deployed version of the application is also accesible through this web app: [imagecaptionai.vercel.app](https://imagecaptioningai.vercel.app/)

## Experiment results
The notebooks in this project covers several experiments with these changes and metrics for each version; each change also includes the changes from the previous versions:
  * **v1:** Initial model, using EfficientNetB2 and Transformers
  * **v2:** Using Bidirectional LSTM instead of Transformers
  * **v3:** Using EfficientNetV2B0 and replacing Bidirectional with vanilla LSTM
  * **v4:** Using a larger Flickr30k dataset, batch size 64 as a workaround for memory overload
  * **v5 and v6:** Added higher regularization and dropout
  * **v7:** Model simplification, reducing the number of LSTM units, attention units, and dense units by half
  * **v8:** Model simplification, reducing the number of LSTM units, attention units, and dense units to half of the previous reduction, using a batch size of 128
  * **v10:** Reverted to v3 with the addition of Exploratory Data Analysis (EDA)

| Model Version | Validation Accuracy | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr\* |
| :-----------: | :-----------------: | :----: | :----: | :----: | :----: | :----: | :-----: | :-----: |
| v1            | 0.3438              | 0.7958 | 0.5476 | 0.3691 | 0.2354 | 0.3174 | 0.4083  | 0.2947  |
| v2            | 0.3460              | 0.5193 | 0.3252 | 0.2097 | 0.1435 | 0.2979 | 0.3613  | 0.2031  |
| v3            | 0.4127              | 0.5951 | 0.4238 | 0.2895 | 0.1995 | 0.3996 | 0.4517  | 0.3013  |
| v4            | 0.3784              | \-     | \-     | \-     | \-     | \-     | \-      | \-      |
| v5            | 0.3612              | \-     | \-     | \-     | \-     | \-     | \-      | \-      |
| v6            | 0.3431              | \-     | \-     | \-     | 0.0505 | 0.1857 | \-      | 0.1632  |
| v7            | 0.3726              | \-     | \-     | \-     | 0.0504 | 0.1876 | \-      | 0.1682  |
| v8            | 0.3404              | \-     | \-     | \-     | \-     | \-     | \-      | \-      |
| v10           | 0.4127              | 0.5951 | 0.4238 | 0.2895 | 0.1995 | 0.3996 | 0.4517  | 0.3013  |

\* Due to the size of the computation and time required, some model versions did not undergo full metric evaluation and are marked with (-).

## How to run the application
### Method A: Run both backend and frontend
#### 1. Run backend (Python)
Create and activate a virtual environment, install Python dependencies, then start the Flask app:
```python
# install requirements
pip install -r requirements.txt

# run backend (adjust path if working dir is backend)
python backend/app.py
```
The backend listens on port 7860 (as configured in app.py).
#### 2. Run frontend (Vite + React)
Install Node dependencies and start the dev server: 
```cmd
cd app\frontend\main-app
npm install
npm run dev
```
Vite dev server will show a localhost URL (usually http://localhost:5173).
#### Notes
* The frontend expects to call the backend API at /predict — if the frontend is configured to call a different host/port, set a proxy or update the frontend API URL to http://localhost:7860.
* If you want to run backend and frontend concurrently, open two PowerShell windows (one for backend, one for frontend).