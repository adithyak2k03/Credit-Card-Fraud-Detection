# Credit-Card-Fraud-Detection
Mini project on Credit card fraud detection using machine learning and deep learning algorithms 

Download the dataset we used here and apply the code given in preprocessing folder. This generates 3 new preprocessed datasets. They are downsampled ,upsampled and imbalanced dataset with preprocessed performed on "Amount" feature.

Downsampling: Since the dataset is highly imbalanced the model cannot learn the pattern of fraudulent cases and will be biased towards majority class. 
Therefore we downsample the majority class by using resample function from sklearn.utils package to bring the count of majority class count to closer to minority class.


Upsampling: We can also solve the problem of imbalanced dataset by using upsampling. In this technique we increase the count of minority class to either to the count of majority class or an arbitrarily chosen number and reduce the majority class to that number of records.
We performed upsampling using the average of every 3 consecutive records and repeat the same for all the minority class records in the dataset.

