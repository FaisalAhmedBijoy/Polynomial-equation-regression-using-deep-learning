from cProfile import label
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
from datasets import polynomail_data
from models.model import define_model_architecture

def model_graph_evaluation(history):
    plt.plot(history.history['loss'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('LOSS')
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Accuracy')
    plt.show()

def model_evaluation(x,y,model):
    results=model.evaluate(x,y)
    print('Accuracy of the model: ',results[1]*100)

    y_pred=model.predict(x)
    plt.scatter(x,y,label='True')
    plt.scatter(x,y_pred,label='Predicted')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(' True vs Predicted')
    plt.show()

   
if __name__=='__main__':
    data_range_select=100 
    x,y=polynomail_data.create_dataset(data_range_select)
 
    print('Data: \n',x)
    print('Label: \n',y)
    x=x/np.max(x)
    y=y/np.max(y)

    model=define_model_architecture()
    model.summary()

    model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
    history=model.fit(x,y,epochs=100)

    saved_model_name='models/polynomial_model.h5'
    model.save(saved_model_name)
    
    model_graph_evaluation(history)
    
    results=model_evaluation(x,y,model)
    
    
    

