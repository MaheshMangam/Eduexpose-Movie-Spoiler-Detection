import gradio as gr
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.layers import Input, Lambda, Dropout, Dense
from tensorflow.keras.models import Model
from keras.saving import register_keras_serializable

# Load tokenizer
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Load BERT model
bert_model = TFBertModel.from_pretrained(MODEL_NAME)

# Register custom Lambda layer function
@register_keras_serializable(package="Custom")
def get_bert_output(inputs):
    input_ids, attention_mask = inputs
    return bert_model(input_ids=input_ids, attention_mask=attention_mask)[1]

# Recreate model structure
input_ids = Input(shape=(128,), dtype=tf.int32, name='input_ids')
attention_mask = Input(shape=(128,), dtype=tf.int32, name='attention_mask')
bert_output = Lambda(get_bert_output, output_shape=(768,))([input_ids, attention_mask])
dropout = Dropout(0.3)(bert_output)
output = Dense(1, activation='sigmoid')(dropout)
model = Model(inputs=[input_ids, attention_mask], outputs=output)

# Load model weights
model.load_weights("bert_model.keras")

# Prediction function
def predict_spoiler(text):
    tokens = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="tf")
    prediction = model.predict([tokens['input_ids'], tokens['attention_mask']])[0][0]

    label = "Spoiler" if prediction > 0.5 else "Not a Spoiler"
    return f"{label} (Probability: {(prediction)*100:.2f}%)"

# Gradio UI
iface = gr.Interface(
    fn=predict_spoiler,
    inputs=gr.Textbox(lines=5, placeholder="Enter movie review..."),
    outputs="text",
    title="Movie Spoiler Detector",
    description="Predicts whether a movie review contains spoilers."
)

iface.launch()


