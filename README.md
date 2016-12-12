# Keras tutorial

##Install for Windows

####Install Anaconda 3
https://www.continuum.io/downloads
choose for Python 3.5
run setup as administrator
choose to add anaconda to path
choose to set python 3.5 as default
other options didn't work for me, reinstalled 3 times...

Another option is just to install python 3.5

####Install using pip:
pip install tensorflow

Install keras:
<i>pip install keras</i>

Configure Keras backend:
https://keras.io/backend/
set backend to "tensorflow"
Theano backend was extremely slow for me on CPU

##Install for Mac OS
Should be no problem, follow the instructions

##Other versions of python installed on the machine

Other versions of python may interfere with Keras / Tensorflow. Either use virual env or docker or config shell scripts

##Test installation
Run cnn_mnist.py example. Training should start.
 