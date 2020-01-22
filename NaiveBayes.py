import pandas as pd
import numpy as np


# Read data into pandas dataframe
data = pd.read_csv('C:/Users/User/Desktop/CS 545/spambase/spambase.data.csv')

# Class accepts pandas dataframe as input
class NB:
    
    def __init__(self, data):
        self.data = data
        # randomly shuffle rows to mix spam/not spam cases
        self.data = data.sample(frac = 1)
        # select half the data for training set
        self.train = data.sample(frac = .5)
        # select half the data for test set
        self.test = data.drop(self.train.index)        
        # convert to numpy arrays
        self.train = np.asarray(self.train)
        self.test = np.asarray(self.test)                
        
    def fit(self):
        # Calculate probability of class
        self.pClassSpam = np.mean(self.train[:,57], axis=0) 
        self.pClassNotSpam = 1 - self.pClassSpam
        
        # Split into Spam/NotSpam
        self.SpamTrainTemp = []
        self.NotSpamTrainTemp = []
        for i in range(len(self.train)):
            if self.train[i,57] == 1:
                self.SpamTrainTemp.append(self.train[i])
            else:
                self.NotSpamTrainTemp.append(self.train[i])
        
        # Convert lists into numpy arrays
        self.SpamTrain = np.asarray(self.SpamTrainTemp)
        self.NotSpamTrain = np.asarray(self.NotSpamTrainTemp)
        
        # Calculate feature means given class, Omit class column
        self.MuSpamTrain = np.mean(self.SpamTrain[:,0:57], axis = 0) 
        self.MuNotSpamTrain = np.mean(self.NotSpamTrain[:,0:57], axis = 0)
        
        #Calculate Standard deviation given class, Omit class column
        self.SigmaSpamTrain = np.std(self.SpamTrain[:,0:57], axis = 0)
        self.SigmaNotSpamTrain = np.std(self.NotSpamTrain[:,0:57], axis = 0)
        
        # Assign minimal standard deviation
        for i in range(len(self.SigmaSpamTrain)):
            if self.SigmaSpamTrain[i] < .0001:
                self.SigmaSpamTrain[i] = .0001
            if self.SigmaNotSpamTrain[i] < .0001:
                self.SigmaNotSpamTrain[i] = .0001
                
    # calculate probability of test x given not spam               
        self.numNotSpam = np.power(np.subtract(self.test[:,0:57],self.MuNotSpamTrain),2) 
        self.denNotSpam = 2*(np.power(self.SigmaNotSpamTrain,2))
        self.ratioNotSpam = np.divide(self.numNotSpam,self.denNotSpam)
        self.expNotSpam = np.exp(-(self.ratioNotSpam))
        self.t1NotSpam = 1/(np.sqrt(2*np.pi)*self.SigmaNotSpamTrain)
        self.pNotSpam =self.t1NotSpam*self.expNotSpam
        
    # calculate probability of test x given spam 
        self.numSpam = np.power(np.subtract(self.test[:,0:57],self.MuSpamTrain),2) 
        self.denSpam = 2*(np.power(self.SigmaSpamTrain,2))
        self.ratioSpam = np.divide(self.numSpam,self.denSpam)
        self.expSpam = np.exp(-(self.ratioSpam))
        self.t1Spam = 1/(np.sqrt(2*np.pi)*self.SigmaSpamTrain)
        self.pSpam = self.t1Spam*self.expSpam

    # Create array of bools to calculate classification accuracy
        self.predClassSpam = np.prod(self.pSpam, axis =1)*self.pClassSpam
        self.predClassNotSpam = np.prod(self.pNotSpam, axis =1)*self.pClassNotSpam
        self.predClass = np.argmax(np.column_stack((self.predClassNotSpam, self.predClassSpam)), axis = 1)
        self.accuracy = np.sum(np.equal(self.predClass,self.test[:,57]))/len(self.test)
        return self.accuracy
   
    # Produce 2x2 confusion matrix
    def cm(self):
        self.conf = np.zeros((2,2))
        for i in range(2):
            for j in range(2):
                self.conf[i,j] = np.sum(np.where(self.predClass==i,1,0)*np.where(self.test[:,57]==j,1,0))
        return self.conf
    
    def acc_metrics(self):
        self.recall = self.conf[1,1]/np.sum(self.conf[:,1])
        print('recall:', self.recall)
        self.precision = self.conf[1,1]/np.sum(self.conf[1,:])
        print('precision:', self.precision)
        

# Fit model to data
model = NB(data)
model.fit()
model.cm()
model.acc_metrics()


model.test[:,57].sum()
2300-1299
858/(858+385)
