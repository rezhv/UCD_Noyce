# UCD_Noyce
## objective
Our overarching objective is to build systems that investigate biases in social media recommendation algorithms especially in politics. To this end, we are interested in measuring political partisanship, toxicity, and hate speech on online social media platforms. Our goals are to: (1) build reliable data collection infrastructure using platform APIs or web scraping tools for Facebook, Reddit, and YouTube; (2) build machine learning models to classify political ideology and other attributes of social media posts. 

## methodology 

We will be using NLP techniques with deep learning including fine tuning transformers to train models that specialize in classifying text based on toxicity. Our current focus at the moment is Bidirectional Encoder Representations from Transformers (BERT). BERT is a neural network transformer model that is pre trained to be able to understand and process language. Our goal is to use our own data to apply transfer learning on a BERT model and make it specialize in detecting toxicity and hate speech on online social media platforms. One of our options is to use the Hugging Face Library which provides pretrained models (e.g. BERT) on a large corpora of text. We can then fine-tune the model for our downstream task. Our team will need to run experiments and find the parameters that work best for our purposes. We will be using web scraping techniques and libraries like Selenium to gather our own data from political websites. 

## methodology 
You can run the train the classifier through the command line. 
First, CLone the repository:
```bash
git clone  https://github.com/rezhv/UCD_Noyce.git
```
Here is a list of possible arguments to pass to the trainer
```bash
python3 /content/UCD_Noyce/Noyce/train_classifier.py -h 
optional arguments:
  -h, --help            show this help message and exit
  -e EPOCHS, --epochs EPOCHS
                        Number of Training Epochs
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch Size
  -l LOGGING_STEPS, --logging_steps LOGGING_STEPS
                        Number of Steps to Run Evaluation
  -r LEARNING_RATE, --learning_rate LEARNING_RATE
                        Value of Learning Rate
```
