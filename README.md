# DATA SCIENCE HACKATHON: INTELLIGENT SEARCH ENGINE
___
## Description
A shortage of skilled labor is one of the evolving problems in
manufacturing. AR technology can speed up workforce training 
in addition to decreasing work-related injuries, cutting 
operational costs, and increasing productivity using digital 
workflows.

## Technologies
Project is created with:
+ `Pillow` version: 7.2.0
+ `matplotlib` version: 3.3.1
+ `numpy` version: 1.18.5
+ `openpyxl` version: 3.0.7
+ `pandas` version: 1.1.1
+ `scikit-learn` version: 0.23.2
+ `tqdm` version: 4.48.2
+ `boto3` version: 1.18.16
+ `gensim` version: 3.8.3
+ `nltk` version: 3.6.2
+ `spacy` version: 2.3.5

## Instructions
1. Clone the repo.
2. (Recommended) Create and activate a virtualenv under the `env/ directory`. Git is already configured to ignore it.
3. Install the very minimal requirements using `pip install -r requirements.txt`
4. Run Jupyter in whatever way works for you. The simplest would be to run `pip install jupyter && jupyter notebook`.
5. Start work.

## Explanation
This project is a system for finding any relevant information
within enterprise documentation.

### Data information
1. Readable and scanned PDF manuals of products

2. For Hackathon purposes, Emerson and Ashcroft products documents are used which include:

   + Measurement Instrumentation
   + Valves, actuators, and regulators
   + Other Sensors and transmitters

3. More than 1,000 documents 

### Metric Information
This model will to be evaluated with `Mean Average Precision at 5 (MAP@5) metric`

### Project Steps
1. **Text Extraction**
    + GCP Vision
2. **Text Preprocessing**
    + Hyphen or Spaces Removing
    + Unuseful Text Removing
    + Paragraphs Joining
    + Other
3. **Dataset Creating** 
    + Split Text into Sentences
    + Split Sentence into Words
    + Identify Pages, Documents Names for Texts
    + Other
4. **Modelling**
    + Text Preprocessing:
      + Stop Words
      + Duplicates
      + Very short words
    + Wor2Vec
    + FastText
    + Normalization 
    + Other
5. **Vector Creating**
   + Extend Dataset with Vectors
   + Grouping Data

6. **Searching similar and metric score in validation data.**

## Author
👤 **Nazarii Drushchak**
+ **Email:** naz2001r@gmail.com
+ **Kaggle:** [naz2001r](https://www.kaggle.com/naz2001r/account)



