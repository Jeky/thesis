import globe
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.metrics import *
from data import *

def crossValidate(dataset, cls, out):
    kfold = KFold(len(dataset.users), 10, shuffle = True, random_state = 42)

    count = 1
    for trainIndex, testIndex in kfold:
        out.write('Cross Validation %d Time\n' % count)
        globe.getLogger().info('Cross Validation %d Time' % count)

        trainX = dataset.instances[trainIndex]
        trainY = dataset.labels[trainIndex]

        testX = dataset.instances[testIndex]
        testY = dataset.labels[testIndex]

        globe.getLogger().info('Training...')
        cls.fit(trainX, trainY)

        globe.getLogger().info('Testing...')
        predicted = cls.predict(testX)

        out.write(np.array_str(confusion_matrix(testY, predicted)) + '\n')

        count += 1


