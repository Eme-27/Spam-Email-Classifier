
Spam Email Classifier

A Python-based machine learning project that classifies emails as spam or notspam using natural language processing.

Features

* Trains a scikit-learn model on your email dataset.

* Predicts if a new email is spam or not.

* Includes unit tests to ensure model and prediction correctness.

* Easy to run and test locally.
  
Installation

1. Clone the repository:
    git clone https://github.com/Eme-27/Spam-Email-Classifier.git
    cd Spam-Email-Classifier/spam-email-classifier
2. Create and activate a virtual environment:
   On Windows CMD:
   
   python -m venv venv
   venv\Scripts\activate.bat
   
   On Windows PowerShell:
   
    python -m venv venv
   .\venv\Scripts\Activate.ps1
3. Install dependencies:
   pip install -r requirements.txt
Run the classifier:
python spam_classifier.py

Running Tests:
python -m unittest discover -s . -p "test_*.py"

You should see output like:
✅ Model trained successfully with accuracy: 100.00%
.
----------------------------------------------------------------------
Ran 4 tests in 0.125s

OK


  

