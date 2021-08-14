# Dog classification webservice
Webservice with an image dog breed classifier

# Overview
With the dog classification webservice, when given an image as input, if it is a dog it will reply with the dog's breed. The classifier is a CNN that was trained in 100 epochs in our train dataset from a pe-trained resnet50 NN from pytorch. The accuracy was 85% in the test dataset.

# Setup
Create a python3 virtualenv::

    python3 -m venv path

Install the libraries in the requirements.txt file::

    pip install -r requirements.txt

Launch the webservice by running the python script **webservice.py**::

    python -m webservice
 
To test the webservice run the **webservice_call** script. Replace the "input" variable with the image path of the image you want to classify::

    python -m webservice_call
    
Check the dog breed of you dog!

# Train the CNN
Run the "train.py" script::

    python -m train
* Make sure you have GPU available or choose a smaller epoch number otherwise it will take days to train 100 epochs


