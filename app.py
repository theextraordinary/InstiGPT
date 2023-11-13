from flask import Flask, render_template, request
import gpt  # Assuming instigpt is the module with your question-answering logic

app = Flask(__name__)

# Use a global variable to store the trained model
trained_model = None

# Train the model only once when the server starts
if trained_model is None:
    trained_model = gpt.train_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/answer_question', methods=['POST'])
def answer_question():
    user_question = request.form['user_question']

    # Use the pre-trained model for answering questions
    answer = gpt.answer_question(user_question, model=trained_model)
    
    return render_template('index.html', user_question=user_question, answer=answer)

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
