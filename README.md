# LT2222 V21 Assignment 2

Assignment 2 from the course LT2222 in the University of Gothenburg's winter 2021 semester.

Your name: Patricia Grau Francitorra

*Answer all questions in the notebook here. You should also write whatever high-level documentation you feel you need to here.*

PART 1 - precprocessing

In part 1, the function preprocess removes all punctuation from the entryfile. The function preprocess2 removes stopwords as well as punctuation, to reduce the size of the feature space. This way, we can see if stopwords are informative, in this case. Results using preprocess2 are saved in the copy of the Jupyter Notebook.

PART 5 - Evaluation

As expected, the training results are better than the testing ones, although not perfect. The classes in which there are more training data (geo, gpe, org, tim) get better results than those classes who have less data to train and test. The class "nat", for example, has only 17 instances (or 16 in the copy of the notebook) in the training data and gets very low results (0 true positives using both preprocess and preprocess2). Classes "art" and "eve" do not have good results either, and quite low training data (~30 instances).

There does not seem to be a big difference in using preprocess or preprocess1, in part 1 of the assignment. Both testing results seems to be quite similar, and differences probably come to the randomness of the data separation in part three (ttsplit).