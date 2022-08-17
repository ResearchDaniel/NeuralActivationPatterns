# Neural Activation Patterns (NAPs): Visual Explainability of Learned Concepts
This is an open-source framework for extracting and interactively visualizing NAPs from pretrained neural networks. A Neural Activation Pattern is a cluster in the activation space of a layer. 
The NAP approach is a novel explainability method for building a fundamental understanding of what neural network layers have learned.


The framework is divided into two parts. The first part, [nap](/nap), finds clusters of activation profiles across different inputs (NAPs).
The second part, [magnifying_glass](/magnifying_glass), is an interactive visualization environment that enables a human-in-the-loop approach to interpret NAPs.


# Installation
Install python requirements via pip:

`python -m pip install -r requirements.txt`

 or conda

 `conda env create -f environment.yml`

The Magnifying glass is a web-based solution built on NodeJS. So, we need to setup a web developer environment:
1. [Download and install NodeJS](https://nodejs.org/en/download/)
2. Install yarn `npm install --global yarn`
3. Install frontend packages
```
cd magnifying_glass/frontend \
yarn install
```
Done!

# Usage
The first step to analyzing NAPs is to compute them using the `nap` library. The second step is to explore them using the Magnifying Glass.

## NAP computation
We provide a convenience script that downloads datasets, computes NAPs, and exports the results to the Magnifying Glass. 
You can adapt the `data_path` to a folder of your liking.
```
# Compute NAPs and export them to magnifying_glass/backend/data
python -m model_analysis.py --model cifar10 --data_set cifar10 --layer dense --data_path ../tensorflow_datasets 
```
**_NOTE:_** The `nap` library caches most of the computations in a results folder to make subsequent runs faster. Subsequent runs might therefore be significantly faster.

## Interactive exploration using the Magnifying Glass
Start the 
From one command prompt, go to the backend folder and start the backend, which serves images to the frontend:
```
cd magnifying_glass/backend \
python server.py
```
From a second command prompt, go to the frontend folder and start the frontend:
```
cd magnifying_glass/frontend \
yarn dev
```
The command line output should point you to a local web address, [http://localhost:8080/](http://localhost:8080/), at which you can explore the exported NAPs, max activations, and more. You simply refresh the Magnifying Glass webpage in case you compute more NAPs.  
