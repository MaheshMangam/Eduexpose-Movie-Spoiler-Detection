## 🎬Eduexpose - Movie Spoiler Detection

This project detects whether a given sentence or review contains a **Movie spoiler** using a fine-tuned **BERT-based NLP model**.  
It was developed during my 2-month internship at **Eduexpose** as part of an AI project.
The app is deployed on **Hugging Face Spaces** using **Gradio** for an interactive UI.

## 🚀 Live Demo

🔗 Try it on Hugging Face
https://huggingface.co/spaces/MaheshMangam/Movie-Spoiler-Detector 

## 🧠 Model Details

- **Base Model**: `bert-base-uncased`
- **Task**: Binary classification (`Spoiler` or `Not Spoiler`)
- **Training Dataset**: IMDB reviews dataset (preprocessed)
- **Model Format**: Saved in `.keras` format and loaded during app runtime

## 🛠️ Tech Stack

| Component     | Tool/Library            |
|---------------|-------------------------|
| Model         | BERT (via Transformers) |
| Interface     | Gradio                  |
| Deployment    | Hugging Face Spaces     |
| Language      | Python                  |

## 📂 Project Structure

├── app.py # Gradio frontend + prediction logic
├── bert_model.keras # Trained BERT model
├── requirements.txt # Python dependencies
├── README.md # Project documentation

## ⚙️ How to Run Locally

1. **Clone the repo**
   ```bash
   git clone https://github.com/MaheshMangam/Eduexpose-Movie-Spoiler-Detection.git
   cd Eduexpose-Movie-Spoiler-Detection
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
3. **Run the app**
   ```bash
   python app.py
   
## 📊 Dataset

Source: IMDB Spoiler/Non-spoiler review dataset

Note: Dataset not included in repo due to size. You can download from:
https://huggingface.co/datasets/imdb

## 📌 Features

✅ Detects whether a text contains a movie spoiler
✅ Clean and interactive UI with Gradio
✅ Easy to deploy or extend with new models or datasets
