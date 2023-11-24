import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("mrm8488/bert-tiny-finetuned-sms-spam-detection")
model = AutoModelForSequenceClassification.from_pretrained("mrm8488/bert-tiny-finetuned-sms-spam-detection")
def classify_spam(text):
    encoded_text = tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    predictions = model(**encoded_text)
    predicted_probabilities = predictions.logits.softmax(dim=1)
    predicted_class = "Spam" if predicted_probabilities[0, 1] > 0.5 else "Not Spam"
    return predicted_class
def main():
    st.title("SMS Spam Classification App")
    st.text("Made by Moneeb Ahmad with Lil Love ❤️  ")
    text_input = st.text_area("Enter SMS text for classification:", "")
    if st.button("Classify"):
        if text_input:
            result = classify_spam(text_input)
            st.subheader("Predicted Class:")
            st.write(result)
        else:
            st.warning("Please enter some text for classification.")

if __name__ == "__main__":
    main()

