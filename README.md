Mini project on Credit card fraud detection using machine learning and deep learning algorithms on a balanced dataset.
Algorithms used Logistic regression, K-nearest neighbours, Support vector machines, Decision trees, Random forest, Xgboost, CNN, and MLP.

Download the dataset we used here and apply the code given in the preprocessing folder. This generates 3 new preprocessed datasets. They are downsampled, upsampled, and an imbalanced dataset with preprocesses performed on the "Amount" feature.

Downsampling: Since the dataset is highly imbalanced the model cannot learn the pattern of fraudulent cases and will be biased towards the majority class. 
Therefore we downsample the majority class by using the resample function from sklearn.utils package to bring the count of majority class count to closer to the minority class.


Upsampling: We can also solve the problem of the imbalanced dataset by using upsampling. In this technique, we increase the count of the minority class to either the count of the majority class or an arbitrarily chosen number and reduce the majority class to that number of records.
We performed upsampling using the average of every 3 consecutive records and repeated the same for all the minority class records in the dataset.


The possibility of error when executing is when using the KNN model in the web app and all models imbalanced dataset because of size constraint of 25Mb of maximum, I wasn't able to upload these files. So execute the KNN model on the imbalanced dataset, generate a PKL file, and put it in an imbalanced folder. For an imbalanced dataset, split X and y to X train, X test, y train, and y test and dump them in a PKL file to the "imbalanced" folder. And change the paths when reading CSV files or PKL files. 
Install required packages such as Streamlit for the web app on your local machine. This should fix all the errors.
