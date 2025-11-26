import gradio as gr
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load saved model
model_path = "saved_sentiment_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    outputs = model(input_ids, attention_mask=attention_mask)
    prediction = torch.argmax(outputs.logits, dim=1).item()

    if prediction == 1:
        return "😊 Positive Review"
    else:
        return "😡 Negative Review"

# Gradio Interface
interface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(label="Enter Movie Review"),
    outputs=gr.Textbox(label="Sentiment"),
    title="IMDB Sentiment Classifier",
    description="This is a simple app built using Gradio and a pre-trained BERT model."
)

interface.launch()
