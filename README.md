# AI Learning Roadmap Repository

Welcome to my **Artificial Intelligence (AI) Learning Roadmap Repository**! This repository documents my journey to master AI, from beginner to expert, using a **trinity approach**: **Books**, **Courses/Tutorials/Papers**, and **Projects**. The roadmap covers foundational concepts, core machine learning, deep learning, specialized AI, and cutting-edge topics like Embedded AI and Embodied AI. Each topic includes curated resources and hands-on projects to build a comprehensive AI portfolio.

This README serves as the central hub, outlining the roadmap, linking to resources, and organizing projects within the repository. Whether you're a beginner or an advanced learner, use this repo as a guide to learn AI systematically and create your own impactful projects.

## Table of Contents
- [Overview](#overview)
- [Roadmap Structure](#roadmap-structure)
  - [Beginner Level: Foundations and Basics](#beginner-level-foundations-and-basics)
  - [Intermediate Level: Core Machine Learning](#intermediate-level-core-machine-learning)
  - [Advanced Level: Deep Learning and Specialized AI](#advanced-level-deep-learning-and-specialized-ai)
  - [Expert Level: Cutting-Edge and Applied AI](#expert-level-cutting-edge-and-applied-ai)
- [Repository Structure](#repository-structure)
- [How to Use This Repository](#how-to-use-this-repository)
- [Contributing](#contributing)
- [License](#license)

## Overview
This repository is designed to:
- Provide a **structured roadmap** for learning AI from scratch to advanced topics.
- Curate high-quality **resources** (books, courses, tutorials, papers) for each topic.
- Showcase **projects** that demonstrate practical AI skills, from simple scripts to complex systems.
- Serve as an open-source resource for the AI community to learn, collaborate, and build upon.

The roadmap is divided into four levels: Beginner, Intermediate, Advanced, and Expert. Each level includes topics with associated resources and projects, stored in dedicated folders within the repo.

## Roadmap Structure

### Beginner Level: Foundations and Basics

#### 1. Introduction to AI
**Objective**: Understand AI’s history, definitions, types (narrow AI, general AI, superintelligence), and ethical implications.
- **Books**:
  - ["Artificial Intelligence: A Guide to Intelligent Systems" by Michael Negnevitsky](https://www.amazon.com/Artificial-Intelligence-Guide-Intelligent-Systems/dp/1408225743) - Beginner-friendly overview of AI concepts.
  - ["AI Superpowers: China, Silicon Valley, and the New World Order" by Kai-Fu Lee](https://www.amazon.com/AI-Superpowers-China-Silicon-Valley/dp/132854639X) - Explores AI’s global impact.
  - ["Life 3.0: Being Human in the Age of Artificial Intelligence" by Max Tegmark](https://www.amazon.com/Life-3-0-Being-Artificial-Intelligence/dp/1101970316) - Discusses AI’s future and ethics.
  - ["The Hundred-Page Machine Learning Book" by Andriy Burkov](https://www.amazon.com/Hundred-Page-Machine-Learning-Book/dp/199957950X) - Concise AI and ML intro.
- **Courses/Tutorials/Papers**:
  - **Course**: [CS50’s Introduction to Artificial Intelligence with Python](https://cs50.harvard.edu/ai/2020/) (Harvard, free) - Covers AI basics with Python.
  - **Course**: [AI For Everyone by Andrew Ng](https://www.coursera.org/learn/ai-for-everyone) (Coursera, free audit) - Non-technical AI overview.
  - **Course**: [Elements of AI](https://www.elementsofai.com/) (University of Helsinki, free) - Broad intro for all audiences.
  - **Course**: [Introduction to AI](https://www.edx.org/course/introduction-to-artificial-intelligence-ai) (Microsoft, free audit) - Practical AI concepts.
  - **Tutorial**: [Introduction to Artificial Intelligence](https://www.datacamp.com/tutorial/artificial-intelligence-introduction) (DataCamp) - Short, beginner-friendly guide.
  - **Tutorial**: [What is Artificial Intelligence?](https://www.simplilearn.com/tutorials/artificial-intelligence-tutorial/what-is-artificial-intelligence) (Simplilearn) - Beginner overview with examples.
  - **Paper**: ["Artificial Intelligence and Its Applications" by A. M. Turing, 1950](https://academic.oup.com/mind/article/LIX/236/433/986238) - Historical context (via academic libraries or JSTOR).
  - **Paper**: ["The AI Revolution: The Road to Superintelligence" by Tim Urban, 2015](https://waitbutwhy.com/2015/01/artificial-intelligence-revolution-1.html) - Accessible article on AI’s potential.
- **Projects** (see `/Beginner/Introduction_to_AI`):
  - Build a **rule-based chatbot** using Python’s `chatterbot` library ([ChatterBot GitHub](https://github.com/gunthercox/ChatterBot)).
  - Create a **markdown summary** of AI types (narrow, general, superintelligence) with real-world examples (stored in `/docs/ai_types.md`).
  - Develop a **quiz game** in Python to test AI concepts (e.g., definitions, history).
  - Write a **blog-style notebook** in Jupyter explaining AI’s ethical challenges (e.g., bias, privacy).

#### 2. Programming Fundamentals for AI
**Objective**: Learn Python programming and libraries like NumPy and Pandas for AI development.
- **Books**:
  - ["Python Crash Course" by Eric Matthes](https://www.amazon.com/Python-Crash-Course-Eric-Matthes/dp/1593279280) - Comprehensive Python guide.
  - ["Automate the Boring Stuff with Python" by Al Sweigart](https://automatetheboringstuff.com/) - Practical Python (free online).
  - ["Learning Python" by Mark Lutz](https://www.amazon.com/Learning-Python-5th-Mark-Lutz/dp/1449355730) - In-depth Python programming.
- **Courses/Tutorials/Papers**:
  - **Course**: [Python for Everybody](https://www.coursera.org/specializations/python) (University of Michigan, free audit) - Beginner Python course.
  - **Course**: [Introduction to Python Programming](https://www.edx.org/course/introduction-to-python-programming) (Georgia Tech, free audit) - Python fundamentals.
  - **Course**: [Python Basics](https://www.coursera.org/learn/python-basics) (University of Michigan, free audit) - Interactive Python learning.
  - **Tutorial**: [Learn Python - Full Course for Beginners](https://www.youtube.com/watch?v=rfscVS0vtbw) (freeCodeCamp, YouTube) - Hands-on Python tutorial.
  - **Tutorial**: [NumPy Tutorial](https://numpy.org/doc/stable/user/absolute_beginners.html) - Official NumPy guide.
  - **Tutorial**: [Pandas Getting Started](https://pandas.pydata.org/docs/getting_started/index.html) - Official Pandas guide.
- **Projects** (see `/Beginner/Programming_Fundamentals`):
  - Build a **simple calculator** in Python for basic arithmetic operations.
  - Create a **data analysis script** using Pandas to explore a CSV dataset (e.g., [Kaggle’s Iris Dataset](https://www.kaggle.com/datasets/uciml/iris)).
  - Develop a **number guessing game** in Python to practice control structures.
  - Implement a **to-do list manager** using Python lists and file I/O.
  - Create a **Jupyter notebook** visualizing NumPy array operations (e.g., matrix multiplication).

#### 3. Mathematics for AI
**Objective**: Master linear algebra, calculus, probability, and optimization for AI.
- **Books**:
  - ["Linear Algebra and Its Applications" by Gilbert Strang](https://www.amazon.com/Linear-Algebra-Its-Applications-5th/dp/032198238X) - Foundational for vectors and matrices.
  - ["Introduction to Probability" by Joseph K. Blitzstein and Jessica Hwang](https://www.amazon.com/Introduction-Probability-Chapman-Statistical-Science/dp/1466575573) - Probability and statistics for AI.
  - ["Calculus Made Easy" by Silvanus P. Thompson](https://www.amazon.com/Calculus-Made-Easy-Silvanus-Thompson/dp/0312185480) - Simplified calculus.
  - ["Mathematics for Machine Learning" by Marc Peter Deisenroth et al.](https://mml-book.github.io/) - Free, AI-focused math book.
- **Courses/Tutorials/Papers**:
  - **Course**: [Mathematics for Machine Learning](https://www.coursera.org/specializations/mathematics-machine-learning) (Imperial College London, free audit) - Covers linear algebra, calculus, and more.
  - **Course**: [Linear Algebra](https://www.khanacademy.org/math/linear-algebra) (Khan Academy, free) - Interactive linear algebra lessons.
  - **Course**: [Probability and Statistics](https://www.edx.org/course/introduction-to-probability) (MIT, free audit) - Comprehensive probability course.
  - **Tutorial**: [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) (3Blue1Brown, YouTube) - Visual linear algebra explanations.
  - **Tutorial**: [Calculus for Machine Learning](https://machinelearningmastery.com/calculus-for-machine-learning/) - Practical calculus guide.
  - **Paper**: ["A Tutorial on Principal Component Analysis" by Jonathon Shlens, 2014](https://arxiv.org/abs/1404.1100) - Math for dimensionality reduction.
  - **Paper**: ["An Introduction to Statistical Learning Theory" by Vapnik, 1999](https://link.springer.com/book/10.1007/978-1-4757-3264-1) - Statistical foundations.
- **Projects** (see `/Beginner/Mathematics_for_AI`):
  - Implement **matrix operations** (e.g., multiplication, inversion) using NumPy.
  - Create a **probability simulator** in Python for coin tosses or dice rolls.
  - Build a **Jupyter notebook** visualizing gradient descent optimization.
  - Develop a **linear algebra playground** to experiment with vector operations.
  - Analyze a dataset’s statistical properties (e.g., mean, variance) using Python.

### Intermediate Level: Core Machine Learning

#### 4. Machine Learning Fundamentals
**Objective**: Understand supervised/unsupervised learning, evaluation metrics, and model challenges.
- **Books**:
  - ["An Introduction to Statistical Learning" by Gareth James et al.](https://www.statlearning.com/) - Free, beginner-friendly ML book.
  - ["Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646) - Practical ML guide.
  - ["Machine Learning Yearning" by Andrew Ng](https://www.amazon.com/Machine-Learning-Yearning-Technical-Strategy/dp/B07B4TR4N6) - Strategy for ML projects.
- **Courses/Tutorials/Papers**:
  - **Course**: [Machine Learning by Andrew Ng](https://www.coursera.org/learn/machine-learning) (Stanford, free audit) - Classic ML course.
  - **Course**: [Introduction to Machine Learning](https://www.edx.org/course/introduction-to-machine-learning) (Google, free audit) - Practical ML intro.
  - **Course**: [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course) (Google, free) - Interactive ML overview.
  - **Tutorial**: [Scikit-Learn Tutorials](https://scikit-learn.org/stable/tutorial/index.html) - Official ML algorithm guides.
  - **Tutorial**: [Kaggle Learn: Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning) - Hands-on ML tutorials.
  - **Tutorial**: [Machine Learning Mastery](https://machinelearningmastery.com/start-here/#getstarted) - Practical ML guides.
  - **Paper**: ["A Few Useful Things to Know About Machine Learning" by Pedro Domingos, 2012](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf) - Practical ML insights.
  - **Paper**: ["The Elements of Statistical Learning" by Hastie et al., 2009](https://web.stanford.edu/~hastie/ElemStatLearn/) - Free, in-depth ML theory.
- **Projects** (see `/Intermediate/ML_Fundamentals`):
  - Build a **spam email classifier** using scikit-learn (dataset: [UCI Spambase](https://archive.ics.uci.edu/ml/datasets/Spambase)).
  - Create a **model comparison notebook** evaluating metrics (accuracy, precision, recall) on a toy dataset.
  - Develop a **binary classification model** for Titanic survival prediction ([Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)).
  - Implement a **regression model** for predicting car prices ([Kaggle Used Cars Dataset](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)).
  - Create a **visualization dashboard** for ML model performance using Plotly.

#### 5. Supervised Learning Algorithms
**Objective**: Implement and understand algorithms like regression, decision trees, and SVMs.
- **Books**:
  - ["Pattern Recognition and Machine Learning" by Christopher Bishop](https://www.amazon.com/Pattern-Recognition-Machine-Learning-Information/dp/0387310738) - Detailed supervised learning.
  - ["Introduction to Machine Learning with Python" by Andreas C. Müller and Sarah Guido](https://www.amazon.com/Introduction-Machine-Learning-Python-Scientists/dp/1449369413) - Scikit-learn focus.
- **Courses/Tutorials/Papers**:
  - **Course**: [Supervised Machine Learning: Regression and Classification](https://www.coursera.org/learn/supervised-machine-learning-regression-classification) (DeepLearning.AI, free audit).
  - **Course**: [Applied Machine Learning in Python](https://www.coursera.org/learn/python-machine-learning) (University of Michigan, free audit).
  - **Course**: [Machine Learning with Python](https://www.coursera.org/learn/machine-learning-with-python) (IBM, free audit).
  - **Tutorial**: [Decision Trees in Scikit-Learn](https://scikit-learn.org/stable/modules/tree.html) - Official guide.
  - **Tutorial**: [SVM with Scikit-Learn](https://scikit-learn.org/stable/modules/svm.html) - Practical SVM tutorial.
  - **Tutorial**: [Kaggle Learn: Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning) - Advanced supervised learning.
  - **Paper**: ["Random Forests" by Leo Breiman, 2001](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) - Foundational ensemble method.
  - **Paper**: ["Support-Vector Networks" by Cortes and Vapnik, 1995](https://link.springer.com/article/10.1007/BF00994018) - SVM introduction.
- **Projects** (see `/Intermediate/Supervised_Learning`):
  - Predict **house prices** using linear regression and random forests ([Kaggle Boston Housing](https://www.kaggle.com/c/boston-housing)).
  - Build a **decision tree classifier** for Iris flower classification ([UCI Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)).
  - Develop an **SVM model** for text classification (e.g., [UCI Sentiment Labelled Sentences](https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences)).
  - Create a **gradient boosting model** using XGBoost for a Kaggle competition dataset.
  - Implement a **model ensemble** combining multiple algorithms (e.g., random forest + SVM) on a dataset.

#### 6. Unsupervised Learning Algorithms
**Objective**: Learn clustering, dimensionality reduction, and anomaly detection.
- **Books**:
  - ["Data Mining: Concepts and Techniques" by Jiawei Han et al.](https://www.amazon.com/Data-Mining-Concepts-Techniques-Management/dp/0128117605) - Covers clustering and association rules.
  - ["Hands-On Unsupervised Learning Using Python" by Ankur A. Patel](https://www.amazon.com/Hands-Unsupervised-Learning-Using-Python/dp/1492035645) - Practical unsupervised methods.
- **Courses/Tutorials/Papers**:
  - **Course**: [Unsupervised Learning, Recommenders, Reinforcement Learning](https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning) (DeepLearning.AI, free audit).
  - **Course**: [Data Clustering Algorithms](https://www.udemy.com/course/data-clustering-algorithms/) (Udemy, paid but often discounted).
  - **Course**: [Machine Learning with Python: Unsupervised Learning](https://www.datacamp.com/courses/machine-learning-with-python-unsupervised-learning) (DataCamp, free trial).
  - **Tutorial**: [Clustering with Scikit-Learn](https://scikit-learn.org/stable/modules/clustering.html) - Official guide.
  - **Tutorial**: [Dimensionality Reduction with PCA](https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60) - Practical PCA tutorial.
  - **Tutorial**: [Anomaly Detection with Python](https://machinelearningmastery.com/anomaly-detection-algorithms-in-python/) - Hands-on guide.
  - **Paper**: ["t-SNE: A New Technique for Visualization" by Laurens van der Maaten, 2008](https://www.jmlr.org/papers/v9/vandermaaten08a.html) - Dimensionality reduction.
  - **Paper**: ["A Density-Based Algorithm for Discovering Clusters" by Ester et al., 1996](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf) - DBSCAN introduction.
- **Projects** (see `/Intermediate/Unsupervised_Learning`):
  - Cluster **customer data** using k-means ([Kaggle Mall Customers](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)).
  - Visualize **high-dimensional data** using PCA and t-SNE ([MNIST Dataset](https://www.kaggle.com/c/digit-recognizer)).
  - Build an **anomaly detection system** for credit card fraud ([Kaggle Credit Card Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)).
  - Implement **hierarchical clustering** on a biological dataset ([UCI Wine Dataset](https://archive.ics.uci.edu/ml/datasets/wine)).
  - Create a **recommender system** using association rules on a retail dataset.

#### 7. Data Preprocessing and Feature Engineering
**Objective**: Master data cleaning, normalization, and feature selection.
- **Books**:
  - ["Feature Engineering for Machine Learning" by Alice Zheng and Amanda Casari](https://www.amazon.com/Feature-Engineering-Machine-Learning-Principles/dp/1491953241) - Practical guide.
  - ["Data Science for Beginners" by Andrew Park](https://www.amazon.com/Data-Science-Beginners-Concepts-Techniques/dp/B08C2H4Q7B) - Covers EDA and preprocessing.
  - ["Practical Statistics for Data Scientists" by Peter Bruce et al.](https://www.amazon.com/Practical-Statistics-Data-Scientists-Essential/dp/149207294X) - Statistical preprocessing.
- **Courses/Tutorials/Papers**:
  - **Course**: [Data Analysis with Python](https://www.coursera.org/learn/data-analysis-with-python) (IBM, free audit).
  - **Course**: [Data Wrangling with Python](https://www.datacamp.com/courses/data-wrangling-python) (DataCamp, free trial).
  - **Course**: [Feature Engineering](https://www.kaggle.com/learn/feature-engineering) (Kaggle, free).
  - **Tutorial**: [Kaggle Learn: Data Cleaning](https://www.kaggle.com/learn/data-cleaning) - Hands-on preprocessing.
  - **Tutorial**: [Feature Engineering Tutorial](https://www.datacamp.com/community/tutorials/feature-engineering-python) - Practical guide.
  - **Tutorial**: [Pandas Data Cleaning](https://realpython.com/python-data-cleaning-pandas/) - Real Python guide.
  - **Paper**: ["Data Preprocessing Techniques for Data Mining" by Salvador García et al., 2016](https://www.sciencedirect.com/science/article/pii/S0957417415006848).
  - **Paper**: ["Feature Selection for Knowledge Discovery" by Huan Liu and Hiroshi Motoda, 2007](https://link.springer.com/book/10.1007/978-3-540-74784-0).
- **Projects** (see `/Intermediate/Data_Preprocessing`):
  - Preprocess the **Titanic dataset** ([Kaggle Titanic](https://www.kaggle.com/c/titanic)) with missing value imputation and encoding.
  - Create an **EDA notebook** visualizing data distributions and correlations ([Kaggle Heart Disease](https://www.kaggle.com/ronitf/heart-disease-uci)).
  - Build a **feature engineering pipeline** for a regression dataset ([Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)).
  - Develop a **data cleaning script** for a messy real-world dataset ([Kaggle Dirty Datasets](https://www.kaggle.com/datasets/competitions/data-science-bowl-2018)).
  - Implement **feature selection** techniques (e.g., correlation analysis, mutual information) on a classification dataset.

### Advanced Level: Deep Learning and Specialized AI

#### 8. Introduction to Neural Networks
**Objective**: Understand neural network basics, backpropagation, and optimization.
- **Books**:
  - ["Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville](https://www.deeplearningbook.org/) - Comprehensive deep learning text (free online).
  - ["Neural Networks and Deep Learning" by Michael Nielsen](http://neuralnetworksanddeeplearning.com/) - Free, beginner-friendly book.
  - ["Deep Learning with Python" by François Chollet](https://www.amazon.com/Deep-Learning-Python-Francois-Chollet/dp/1617294438) - Practical guide.
- **Courses/Tutorials/Papers**:
  - **Course**: [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) (DeepLearning.AI, free audit).
  - **Course**: [Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning) (DeepLearning.AI, free audit).
  - **Course**: [Introduction to Deep Learning](https://www.coursera.org/learn/intro-to-deep-learning) (HSE University, free audit).
  - **Tutorial**: [Neural Networks from Scratch](https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5sCgbqsD4A06do0aVLT) (sentdex, YouTube).
  - **Tutorial**: [Deep Learning with Keras](https://keras.io/getting_started/intro_to_keras_for_engineers/) - Official Keras guide.
  - **Tutorial**: [PyTorch Tutorials](https://pytorch.org/tutorials/) - Official PyTorch guide.
  - **Paper**: ["Gradient-Based Learning Applied to Document Recognition" by Yann LeCun et al., 1998](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) - Foundational neural network paper.
  - **Paper**: ["A Neural Algorithm of Artistic Style" by Gatys et al., 2015](https://arxiv.org/abs/1508.06576) - Neural style transfer intro.
- **Projects** (see `/Advanced/Neural_Networks`):
  - Build a **feedforward neural network** for MNIST digit classification ([MNIST Dataset](https://www.kaggle.com/c/digit-recognizer)).
  - Create a **backpropagation visualizer** in Python to show gradient updates.
  - Develop a **neural network from scratch** (without frameworks) for a simple dataset.
  - Implement a **neural style transfer** model for image transformation.
  - Create a **Jupyter notebook** comparing optimization algorithms (e.g., SGD, Adam).

#### 9. Deep Learning Frameworks
**Objective**: Use TensorFlow/Keras or PyTorch for CNNs and RNNs.
- **Books**:
  - ["Deep Learning with Python" by François Chollet](https://www.amazon.com/Deep-Learning-Python-Francois-Chollet/dp/1617294438) - Keras-focused guide.
  - ["Programming PyTorch for Deep Learning" by Ian Pointer](https://www.amazon.com/Programming-PyTorch-Deep-Learning-Applications/dp/1492045357) - PyTorch for practitioners.
  - ["Deep Learning for Coders with fastai and PyTorch" by Jeremy Howard and Sylvain Gugger](https://www.amazon.com/Deep-Learning-Coders-fastai-PyTorch/dp/1492045527) - Practical deep learning.
- **Courses/Tutorials/Papers**:
  - **Course**: [Practical Deep Learning for Coders](https://course.fast.ai/) (fast.ai, free) - Hands-on deep learning.
  - **Course**: [Deep Learning with PyTorch](https://www.udacity.com/course/deep-learning-pytorch--ud188) (Udacity, free).
  - **Course**: [TensorFlow Developer Certificate](https://www.coursera.org/professional-certificates/tensorflow-in-practice) (Google, free audit).
  - **Tutorial**: [TensorFlow Tutorials](https://www.tensorflow.org/tutorials) - Official TensorFlow guides.
  - **Tutorial**: [PyTorch 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) - Quick PyTorch intro.
  - **Tutorial**: [Keras with TensorFlow](https://www.tensorflow.org/guide/keras) - Official Keras guide.
  - **Paper**: ["Deep Learning with PyTorch: A 60 Minute Blitz" by PyTorch Team](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) - Tutorial-style paper.
  - **Paper**: ["Convolutional Neural Networks for Visual Recognition" by Krizhevsky et al., 2012](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) - AlexNet introduction.
- **Projects** (see `/Advanced/Deep_Learning_Frameworks`):
  - Build a **CNN** for image classification ([CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)).
  - Develop an **RNN** for text generation using LSTM ([Shakespeare Dataset](https://www.kaggle.com/datasets/hocinebendou/shakespeare-plays)).
  - Create a **transfer learning model** using pre-trained ResNet for image classification.
  - Implement a **sequence-to-sequence model** for machine translation (e.g., English to French).
  - Build a **web app** to demo a deep learning model using Flask.

#### 10. Reinforcement Learning
**Objective**: Learn RL concepts and algorithms like Q-learning and DQN.
- **Books**:
  - ["Reinforcement Learning: An Introduction" by Richard S. Sutton and Andrew G. Barto](http://incompleteideas.net/book/the-book-2nd.html) - Standard RL text (free online).
  - ["Deep Reinforcement Learning Hands-On" by Maxim Lapan](https://www.amazon.com/Deep-Reinforcement-Learning-Hands-Second/dp/1838826998) - Practical RL with PyTorch.
- **Courses/Tutorials/Papers**:
  - **Course**: [Reinforcement Learning Specialization](https://www.coursera.org/specializations/reinforcement-learning) (University of Alberta, free audit).
  - **Course**: [Deep Reinforcement Learning](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) (Udacity, free audit for some content).
  - **Course**: [Fundamentals of Reinforcement Learning](https://www.coursera.org/learn/fundamentals-of-reinforcement-learning) (University of Alberta, free audit).
  - **Tutorial**: [OpenAI Gym Tutorials](https://gym.openai.com/docs/) - Practical RL environments.
  - **Tutorial**: [Reinforcement Learning with Stable-Baselines](https://stable-baselines3.readthedocs.io/en/master/) - Practical RL library guide.
  - **Tutorial**: [RL Tutorial by Arthur Juliani](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0) - Q-learning intro.
  - **Paper**: ["Playing Atari with Deep Reinforcement Learning" by Mnih et al., 2013](https://arxiv.org/abs/1312.5602) - DQN foundational paper.
  - **Paper**: ["Deep Reinforcement Learning with Double Q-Learning" by van Hasselt et al., 2015](https://arxiv.org/abs/1509.06461) - Advanced RL.
- **Projects** (see `/Advanced/Reinforcement_Learning`):
  - Implement **Q-learning** for FrozenLake ([OpenAI Gym](https://gym.openai.com/envs/FrozenLake-v0/)).
  - Build a **DQN agent** for CartPole ([OpenAI Gym](https://gym.openai.com/envs/CartPole-v1/)).
  - Develop a **policy gradient model** for a simple game environment.
  - Create a **RL agent** for a custom grid-world environment.
  - Implement a **multi-agent RL system** for a competitive game (e.g., Pong).

#### 11. Natural Language Processing (NLP)
**Objective**: Master tokenization, embeddings, transformers, and LLMs.
- **Books**:
  - ["Natural Language Processing with Python" by Steven Bird et al.](https://www.amazon.com/Natural-Language-Processing-Python-Analyzing/dp/0596516495) - NLTK-based NLP basics.
  - ["Speech and Language Processing" by Dan Jurafsky and James H. Martin](https://web.stanford.edu/~jurafsky/slp3/) - Comprehensive NLP text (free draft).
  - ["Deep Learning for Natural Language Processing" by Palash Goyal et al.](https://www.amazon.com/Deep-Learning-Natural-Language-Processing/dp/1838987312) - Advanced NLP.
- **Courses/Tutorials/Papers**:
  - **Course**: [Natural Language Processing Specialization](https://www.coursera.org/specializations/natural-language-processing) (DeepLearning.AI, free audit).
  - **Course**: [NLP with Python](https://www.udemy.com/course/nlp-natural-language-processing-with-python/) (Udemy, paid but often discounted).
  - **Course**: [Hugging Face NLP Course](https://huggingface.co/course) (Hugging Face, free) - Transformer-focused.
  - **Tutorial**: [NLTK Tutorial](https://www.nltk.org/book/) - Official NLTK guide.
  - **Tutorial**: [Transformers with Hugging Face](https://huggingface.co/docs/transformers/index) - Practical transformer guide.
  - **Tutorial**: [SpaCy Advanced NLP](https://spacy.io/usage/spacy-101) - Advanced NLP library guide.
  - **Paper**: ["Attention is All You Need" by Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) - Transformer model introduction.
  - **Paper**: ["BERT: Pre-training of Deep Bidirectional Transformers" by Devlin et al., 2018](https://arxiv.org/abs/1810.04805) - BERT introduction.
- **Projects** (see `/Advanced/NLP`):
  - Build a **sentiment analysis model** using Hugging Face transformers ([IMDb Dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)).
  - Develop a **text summarization model** using BERT ([CNN/DailyMail Dataset](https://huggingface.co/datasets/cnn_dailymail)).
  - Create a **named entity recognition (NER)** system with SpaCy ([CoNLL-2003 Dataset](https://huggingface.co/datasets/conll2003)).
  - Implement a **chatbot** using a pre-trained LLM (e.g., GPT-2).
  - Build a **machine translation model** for English to Spanish ([WMT Dataset](https://huggingface.co/datasets/wmt14)).

#### 12. Computer Vision
**Objective**: Learn image processing, object detection, and generative models.
- **Books**:
  - ["Computer Vision: Algorithms and Applications" by Richard Szeliski](http://szeliski.org/Book/) - Comprehensive CV text (free online).
  - ["Deep Learning for Computer Vision with Python" by Adrian Rosebrock](https://www.amazon.com/Deep-Learning-Computer-Vision-Python/dp/0982366183) - Practical CV guide.
  - ["Learning OpenCV 4" by Adrian Kaehler and Gary Bradski](https://www.amazon.com/Learning-OpenCV-Computer-Vision-Library/dp/1492034169) - OpenCV focus.
- **Courses/Tutorials/Papers**:
  - **Course**: [Convolutional Neural Networks](https://www.coursera.org/learn/convolutional-neural-networks) (DeepLearning.AI, free audit).
  - **Course**: [Computer Vision Basics](https://www.coursera.org/learn/computer-vision-basics) (University at Buffalo, free audit).
  - **Course**: [Deep Learning for Computer Vision](https://www.udacity.com/course/deep-learning-for-computer-vision--ud810) (Udacity, free audit).
  - **Tutorial**: [OpenCV Tutorials](https://docs.opencv.org/master/d9/df8/tutorial_root.html) - Image processing basics.
  - **Tutorial**: [YOLO Object Detection](https://pjreddie.com/darknet/yolo/) - Official YOLO guide.
  - **Tutorial**: [GANs with PyTorch](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) - Practical GAN guide.
  - **Paper**: ["Generative Adversarial Nets" by Goodfellow et al., 2014](https://arxiv.org/abs/1406.2661) - GAN introduction.
  - **Paper**: ["YOLO: Real-Time Object Detection" by Redmon et al., 2016](https://arxiv.org/abs/1506.02640) - YOLO introduction.
- **Projects** (see `/Advanced/Computer_Vision`):
  - Build an **object detection model** using YOLO ([COCO Dataset](https://cocodataset.org/)).
  - Develop a **face recognition system** using OpenCV ([Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/)).
  - Create a **GAN** for synthetic image generation ([CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)).
  - Implement an **image segmentation model** using U-Net ([Kaggle Carvana Image Masking](https://www.kaggle.com/c/carvana-image-masking-challenge)).
  - Build a **real-time webcam-based object detector** using TensorFlow or PyTorch.

### Expert Level: Cutting-Edge and Applied AI

#### 13. Generative AI and Advanced Models
**Objective**: Explore diffusion models, VAEs, and multimodal AI.
- **Books**:
  - ["Generative Deep Learning" by David Foster](https://www.amazon.com/Generative-Deep-Learning-Teaching-Machines/dp/1492041947) - Covers GANs, VAEs, and more.
  - ["Deep Learning for Coders with fastai and PyTorch" by Jeremy Howard and Sylvain Gugger](https://www.amazon.com/Deep-Learning-Coders-fastai-PyTorch/dp/1492045527) - Advanced generative models.
- **Courses/Tutorials/Papers**:
  - **Course**: [Generative AI with Large Language Models](https://www.coursera.org/learn/generative-ai-with-llms) (DeepLearning.AI, free audit).
  - **Course**: [Generative Adversarial Networks (GANs)](https://www.coursera.org/learn/generative-adversarial-networks-gans) (DeepLearning.AI, free audit).
  - **Course**: [Diffusion Models](https://www.coursera.org/learn/diffusion-models) (DeepLearning.AI, free audit).
  - **Tutorial**: [Stable Diffusion Tutorial](https://huggingface.co/docs/diffusers) - Diffusion model guide.
  - **Tutorial**: [Variational Autoencoders with PyTorch](https://pytorch.org/tutorials/beginner/vae.html) - Practical VAE guide.
  - **Tutorial**: [Multimodal AI with CLIP](https://openai.com/blog/clip/) - OpenAI’s CLIP guide.
  - **Paper**: ["Denoising Diffusion Probabilistic Models" by Ho et al., 2020](https://arxiv.org/abs/2006.11239) - Diffusion model foundation.
  - **Paper**: ["Variational Autoencoders" by Kingma and Welling, 2013](https://arxiv.org/abs/1312.6114) - VAE introduction.
- **Projects** (see `/Expert/Generative_AI`):
  - Fine-tune a **diffusion model** for image generation ([Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion)).
  - Build a **VAE** for image reconstruction ([MNIST Dataset](https://www.kaggle.com/c/digit-recognizer)).
  - Develop a **multimodal AI model** combining text and images using CLIP.
  - Create a **text-to-image generator** using DALL-E Mini ([Hugging Face DALL-E Mini](https://huggingface.co/spaces/dalle-mini/dalle-mini)).
  - Implement a **music generation model** using GANs or transformers ([MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro)).

#### 14. AI in Production
**Objective**: Deploy models, understand MLOps, and manage data pipelines.
- **Books**:
  - ["Building Machine Learning Powered Applications" by Emmanuel Ameisen](https://www.amazon.com/Building-Machine-Learning-Powered-Applications/dp/149204511X) - Practical deployment guide.
  - ["MLOps: Continuous Delivery and Automation Pipelines in Machine Learning" by Mark Treveil et al.](https://www.amazon.com/Machine-Learning-Engineering-Action-Treveil/dp/1617298719) - MLOps focus.
  - ["Designing Machine Learning Systems" by Chip Huyen](https://www.amazon.com/Designing-Machine-Learning-Systems-Production/dp/1098107969) - End-to-end ML systems.
- **Courses/Tutorials/Papers**:
  - **Course**: [MLOps Specialization](https://www.coursera.org/specializations/mlops-machine-learning-duplicate) (DeepLearning.AI, free audit).
  - **Course**: [Machine Learning Engineering for Production](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops) (DeepLearning.AI, free audit).
  - **Course**: [Deploying Machine Learning Models](https://www.udemy.com/course/deployment-of-machine-learning-models/) (Udemy, paid but often discounted).
  - **Tutorial**: [Deploying ML Models with Flask](https://www.datacamp.com/tutorial/flask-machine-learning) - Practical guide.
  - **Tutorial**: [Docker for Data Scientists](https://towardsdatascience.com/docker-for-data-scientists-5732501f0ba4) - Docker for ML deployment.
  - **Tutorial**: [MLOps with Kubeflow](https://www.kubeflow.org/docs/) - Official Kubeflow guide.
  - **Paper**: ["Hidden Technical Debt in Machine Learning Systems" by Sculley et al., 2015](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf) - MLOps challenges.
  - **Paper**: ["MLOps: Continuous Delivery and Automation Pipelines" by Kreuzberger et al., 2022](https://arxiv.org/abs/2205.12352) - Modern MLOps.
- **Projects** (see `/Expert/AI_in_Production`):
  - Deploy a **classification model** using Flask and Docker ([Heroku](https://www.heroku.com/)).
  - Create an **end-to-end ML pipeline** with data ingestion, training, and deployment ([Kaggle Datasets](https://www.kaggle.com/datasets)).
  - Build a **model monitoring dashboard** using Streamlit to track performance.
  - Implement a **CI/CD pipeline** for ML models using GitHub Actions.
  - Develop a **scalable API** for a deep learning model using FastAPI.

#### 15. AI Ethics, Bias, and Society
**Objective**: Understand fairness, explainability, and societal impacts of AI.
- **Books**:
  - ["Weapons of Math Destruction" by Cathy O’Neil](https://www.amazon.com/Weapons-Math-Destruction-Increases-Inequality/dp/0553418815) - Explores AI bias and ethics.
  - ["The Ethical Algorithm" by Michael Kearns and Aaron Roth](https://www.amazon.com/Ethical-Algorithm-Science-Socially-Design/dp/0190948205) - Technical and ethical balance.
  - ["AI Ethics" by Mark Coeckelbergh](https://www.amazon.com/AI-Ethics-MIT-Press-Essential/dp/0262538199) - Philosophical perspective.
- **Courses/Tutorials/Papers**:
  - **Course**: [AI Ethics](https://www.coursera.org/learn/ai-ethics) (LearnQuest, free audit).
  - **Course**: [Ethics of Artificial Intelligence](https://www.edx.org/course/ethics-of-artificial-intelligence) (Columbia University, free audit).
  - **Course**: [Responsible AI](https://www.coursera.org/learn/responsible-ai) (Google, free audit).
  - **Tutorial**: [FairML Tutorials](https://fairmlbook.org/) - Free fairness tutorials.
  - **Tutorial**: [Explainable AI with SHAP](https://shap.readthedocs.io/en/latest/) - Official SHAP guide.
  - **Tutorial**: [AI Fairness 360](https://aif360.mybluemix.net/) - IBM’s fairness toolkit.
  - **Paper**: ["Fairness and Machine Learning" by Barocas et al., 2019](https://fairmlbook.org/) - Comprehensive fairness guide.
  - **Paper**: ["The Social Dilemma of Autonomous Vehicles" by Bonnefon et al., 2016](https://science.sciencemag.org/content/352/6293/1573) - Ethical dilemmas.
- **Projects** (see `/Expert/AI_Ethics`):
  - Analyze **bias in a dataset** (e.g., [COMPAS Dataset](https://www.propublica.org/datastore/dataset/compas-recidivism-risk-score-data-and-analysis)).
  - Create an **explainable AI model** using SHAP or LIME ([Kaggle Heart Disease](https://www.kaggle.com/ronitf/heart-disease-uci)).
  - Develop a **fairness audit tool** to evaluate model bias.
  - Build a **Jupyter notebook** discussing ethical implications of a real-world AI system.
  - Create a **visualization dashboard** for model interpretability metrics.

#### 16. Embedded AI
**Objective**: Deploy AI on resource-constrained devices using optimization techniques.
- **Books**:
  - ["TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers" by Pete Warden and Daniel Situnayake](https://www.amazon.com/TinyML-Machine-Learning-TensorFlow-Microcontrollers/dp/1492052043) - TinyML guide.
  - ["Embedded Systems with ARM Cortex-M Microcontrollers in Assembly Language and C" by Yifeng Zhu](https://www.amazon.com/Embedded-Systems-ARM-Cortex-M-Microcontrollers/dp/0982692668) - Hardware context.
- **Courses/Tutorials/Papers**:
  - **Course**: [TinyML](https://www.edx.org/course/fundamentals-of-tinyml) (Harvard, free) - Intro to TinyML.
  - **Course**: [Embedded Machine Learning](https://www.coursera.org/learn/embedded-machine-learning) (Edge Impulse, free audit).
  - **Course**: [TinyML and Efficient Deep Learning](https://www.edx.org/course/tinyml-and-efficient-deep-learning-computing) (MIT, free audit).
  - **Tutorial**: [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers) - Official guide.
  - **Tutorial**: [Edge Impulse Tutorials](https://docs.edgeimpulse.com/docs/) - Practical embedded AI.
  - **Tutorial**: [Model Optimization with TensorFlow](https://www.tensorflow.org/model_optimization) - Quantization and pruning guide.
  - **Paper**: ["EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" by Tan and Le, 2019](https://arxiv.org/abs/1905.11946) - Optimization techniques.
  - **Paper**: ["MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" by Howard et al., 2017](https://arxiv.org/abs/1704.04861) - Mobile AI.
- **Projects** (see `/Expert/Embedded_AI`):
  - Deploy a **lightweight CNN** on a Raspberry Pi for image classification ([CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)).
  - Build a **TinyML model** for motion detection using Arduino ([Edge Impulse Datasets](https://docs.edgeimpulse.com/docs/pre-built-datasets)).
  - Implement **model quantization** for a pre-trained model using TensorFlow Lite.
  - Create a **real-time sensor-based classifier** for IoT devices (e.g., temperature anomaly detection).
  - Develop a **keyword spotting system** for voice commands on microcontrollers ([Speech Commands Dataset](https://www.tensorflow.org/datasets/catalog/speech_commands)).

#### 17. Embodied AI
**Objective**: Integrate AI with physical systems like robots for real-world interaction.
- **Books**:
  - ["Probabilistic Robotics" by Sebastian Thrun et al.](https://www.amazon.com/Probabilistic-Robotics-Intelligent-Autonomous-Agents/dp/0262201623) - Robotics and AI integration.
  - ["Embodied Artificial Intelligence" by Rolf Pfeifer and Christian Scheier](https://www.amazon.com/Embodied-Artificial-Intelligence-International-Understanding/dp/354022484X) - Theoretical foundations.
  - ["Robotics, Vision and Control" by Peter Corke](https://www.amazon.com/Robotics-Vision-Control-Fundamental-Algorithms/dp/3319544136) - Practical robotics.
- **Courses/Tutorials/Papers**:
  - **Course**: [Robotics Specialization](https://www.coursera.org/specializations/robotics) (University of Pennsylvania, free audit).
  - **Course**: [Modern Robotics](https://www.coursera.org/specializations/modernrobotics) (Northwestern University, free audit).
  - **Course**: [Self-Driving Cars](https://www.coursera.org/specializations/self-driving-cars) (University of Toronto, free audit).
  - **Tutorial**: [ROS Tutorials](http://wiki.ros.org/ROS/Tutorials) - Robot Operating System guide.
  - **Tutorial**: [Gazebo Simulation Tutorials](http://gazebosim.org/tutorials) - Robotics simulation.
  - **Tutorial**: [Sim-to-Real Transfer Learning](https://towardsdatascience.com/sim-to-real-transfer-learning-for-robotic-applications-8c6b4b6b6f6b) - Practical guide.
  - **Paper**: ["Sim-to-Real Transfer for Robotic Control with Dynamics Randomization" by Peng et al., 2018](https://arxiv.org/abs/1710.06537).
  - **Paper**: ["Learning Dexterous In-Hand Manipulation" by OpenAI, 2018](https://arxiv.org/abs/1808.00177) - Embodied AI in robotics.
- **Projects** (see `/Expert/Embodied_AI`):
  - Build a **robotic arm controller** using RL in a simulated environment ([MuJoCo](http://www.mujoco.org/)).
  - Create a **sim-to-real transfer learning** project for a robot navigation task ([Gazebo Datasets](http://gazebosim.org/)).
  - Develop a **sensor fusion model** for a robotic system using ROS.
  - Implement a **human-robot interaction system** with computer vision and NLP.
  - Build a **SLAM (Simultaneous Localization and Mapping)** system for a mobile robot ([TurtleBot3](https://www.turtlebot.com/)).

#### 18. Emerging Topics and Research
**Objective**: Explore federated learning, quantum AI, multi-agent systems, and AGI pursuits.
- **Books**:
  - ["Federated Learning" by Qiang Yang et al.](https://www.amazon.com/Federated-Learning-Synthesis-Lectures-Artificial/dp/1681736977) - Distributed AI systems.
  - ["Quantum Machine Learning" by Peter Wittek](https://www.amazon.com/Quantum-Machine-Learning-Computing-Artificial/dp/0128100400) - Quantum AI intro.
  - ["Artificial General Intelligence" by Ben Goertzel and Cassio Pennachin](https://www.amazon.com/Artificial-General-Intelligence-Cognitive-Technologies/dp/354023733X) - AGI concepts.
- **Courses/Tutorials/Papers**:
  - **Course**: [Federated Learning](https://www.coursera.org/learn/federated-learning) (DeepLearning.AI, free audit).
  - **Course**: [Quantum Machine Learning](https://www.edx.org/course/quantum-machine-learning) (University of Toronto, free audit).
  - **Course**: [Multi-Agent Systems](https://www.coursera.org/learn/multi-agent-systems) (University of Maryland, free audit).
  - **Tutorial**: [Quantum Machine Learning Tutorials](https://pennylane.ai/qml/) - Free quantum AI resource.
  - **Tutorial**: [Federated Learning with TensorFlow](https://www.tensorflow.org/federated) - Official guide.
  - **Tutorial**: [Multi-Agent RL with PettingZoo](https://www.pettingzoo.ml/) - Multi-agent RL guide.
  - **Paper**: ["A Survey on Multi-Agent Reinforcement Learning" by Zhang et al., 2021](https://arxiv.org/abs/2108.04757).
  - **Paper**: ["Federated Learning: Challenges, Methods, and Future Directions" by Kairouz et al., 2019](https://arxiv.org/abs/1908.07873).
  - **Paper**: ["Quantum Machine Learning" by Biamonte et al., 2017](https://arxiv.org/abs/1611.09347).
- **Projects** (see `/Expert/Emerging_Topics`):
  - Implement a **federated learning model** for a distributed dataset ([Flower Framework](https://flower.dev/)).
  - Create a **quantum ML model** using PennyLane ([Qiskit Datasets](https://qiskit.org/documentation/)).
  - Build a **multi-agent RL system** for a cooperative game ([PettingZoo Environments](https://www.pettingzoo.ml/)).
  - Develop a **literature review notebook** summarizing recent AI papers from arXiv.
  - Create a **sustainability-focused AI model** for energy consumption prediction ([UCI Energy Dataset](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)).

## Repository Structure
The repository is organized as follows:
```
AI_Learning_Roadmap/
├── Beginner/
│   ├── Introduction_to_AI/
│   │   ├── docs/ (summaries, notes)
│   │   ├── projects/ (chatbot, quiz game)
│   ├── Programming_Fundamentals/
│   │   ├── docs/ (tutorials, summaries)
│   │   ├── projects/ (calculator, data analysis)
│   ├── Mathematics_for_AI/
│   │   ├── docs/ (math notes, visualizations)
│   │   ├── projects/ (matrix operations, simulators)
├── Intermediate/
│   ├── ML_Fundamentals/
│   ├── Supervised_Learning/
│   ├── Unsupervised_Learning/
│   ├── Data_Preprocessing/
├── Advanced/
│   ├── Neural_Networks/
│   ├── Deep_Learning_Frameworks/
│   ├── Reinforcement_Learning/
│   ├── NLP/
│   ├── Computer_Vision/
├── Expert/
│   ├── Generative_AI/
│   ├── AI_in_Production/
│   ├── AI_Ethics/
│   ├── Embedded_AI/
│   ├── Embodied_AI/
│   ├── Emerging_Topics/
├── docs/ (general documentation, resource lists)
├── LICENSE
├── README.md
```

Each topic folder contains:
- **docs/**: Notes, summaries, and resource links.
- **projects/**: Jupyter notebooks, Python scripts, and READMEs explaining each project.

## How to Use This Repository
1. **Follow the Roadmap**: Start with the Beginner level and progress through topics in order, or jump to areas of interest.
2. **Explore Resources**: Use the linked books, courses, tutorials, and papers to deepen your understanding.
3. **Build Projects**: Replicate or modify the projects in each topic folder to gain hands-on experience. All datasets and tools are linked.
4. **Contribute**: Add your own projects, improve documentation, or suggest new resources via pull requests.
5. **Track Progress**: Use the `/docs/progress_tracker.md` to mark completed topics and projects.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Add your projects, resources, or documentation.
4. Commit changes with clear messages (`git commit -m "Added new NLP project"`).
5. Push to your branch (`git push origin feature/your-feature`).
6. Open a pull request with a description of your changes.

Please follow the [Code of Conduct](CODE_OF_CONDUCT.md) and ensure projects include clear documentation.

## License
This repository is licensed under the [MIT License](LICENSE). Feel free to use, modify, and share the content, provided you give appropriate credit.
