# Importing Libraries
from flask import render_template, request
from summarizerapp import app
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Loading model
summarization_model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn')
summarization_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn')

# Defining default route
@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

# Route for displaying summary
@app.route("/prediction", methods=['POST'])
def prediction():
    # Getting user input
    text_input = request.form["textinput"]
    min_len = int(request.form["minlen"])
    max_len = int(request.form["maxlen"])
    summarization_method = request.form["summarization_method"]

    # Generating summary
    if summarization_method == "dont_use_given_text":
        summary = bart_summarize(text_input, int(1), float(2.0), max_len, min_len, int(3))  
    else:
        summary = bart_summarize(text_input, int(10), float(2.0), max_len, min_len, int(3))
    
    return render_template('output.html', summary = summary, text_input = text_input, min_len=min_len, max_len=max_len, summarization_method=summarization_method)

# Function to summarize text
def bart_summarize(text, num_beams, length_penalty, max_length, min_length, no_repeat_ngram_size):
    torch_device = 'cpu'
    text = text.replace('\n', '')
    text_input_ids = summarization_tokenizer.batch_encode_plus([text], return_tensors='pt', max_length=1024)[
        'input_ids'].to(torch_device)
    summary_ids = summarization_model.generate(text_input_ids, num_beams=int(num_beams),
                                               length_penalty=float(length_penalty), max_length=int(max_length),
                                               min_length=int(min_length),
                                               no_repeat_ngram_size=int(no_repeat_ngram_size))
    summary_txt = summarization_tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
    return summary_txt