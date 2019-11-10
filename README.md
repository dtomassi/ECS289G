# ECS289G
The python file, BERTS_parts prepares the data in the format that BERT accepts.

run.sh is the script file to run BERT.

BERT code is available here, https://github.com/google-research/bert.git.

BERT model files are available online. They are:
1. BERT-based, uncased
2. BERT-based, cased
3. BERT-Large, uncased
4. BERT-Large, cased

If you have access to Google TPU use the "large" model files.

Put your data into the format BERT expects. Create a folder in the directory where you cloned BERT. You’ll be adding three separate files there called train.tsv dev.tsvand test.tsv (tsv, for tab separated values). In train.tsv and dev.tsv you should have four columns with no headers as follows:

Column 1: An ID for the row (can be just a count, or even just the same number or letter for every row, if you don’t care to keep track of each individual example).

Column 2: A label for the row as an int. These are the classification labels that your classifier aims to predict.

Column 3: A column of all the same letter — this is a throw-away column that you need to include because the BERT model expects it.

Column 4: The text examples you want to classify.

Here is an example of what the data in train.tsv and dev.tsvshould look like:

1    0    a    an example of text that should fit in class 0

2    1    a    an example of text that should fit in class 1

3    0    a    another class 0 example

4    2    a    a class 2 example


test.tsv should have a slightly different format.


Column 1: an ID for each example, similar to column 1 in the train and dev files, and

Column 2: the text you want to classify. Also,test.tsv should have a header line (whereas train and dev should not). Here is an example of what test.tsv should look like:

id  sentence

1   my first test example

2   another test example. Yay this is fun!

3   yet another test example


