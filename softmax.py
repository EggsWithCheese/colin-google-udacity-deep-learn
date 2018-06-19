"""Softmax."""
import numpy as np


og_scores = [3.0, 1.0, 0.2]

one_d_scores = [1.0, 2.0, 3.0]

two_d_scores = np.array([[1, 2, 3, 6],
                  [2, 4, 5, 6],
                  [3, 8, 7, 6]])
                  
                  
two_d_increasing = np.array([[0.1, 1, 10, 100, 600],
                             [0.2, 2, 20, 200, 600],
                             [0.3, 3, 30, 300, 601]])                  
                   

scores =two_d_increasing

scores = np.array(scores).astype(np.float64)
multiplier=1
scores=scores*multiplier

def column(matrix, i):
    return [row[i] for row in matrix]
    

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x = np.array(x).astype(np.float64)
    # print("x", x)
    
    
    # print("values",values)
    
    
    
    
    #divide 
    softmax = np.exp(x)/np.sum(np.exp(x), axis=0)
    # print("softmax",softmax)
    
    return softmax



# print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt

def bar_chart(labels, values):
    y_pos = np.arange(len(labels))
    plt.bar(y_pos, values, align='center', alpha=0.5)
    plt.xticks(y_pos, labels)
    plt.ylabel('probability')
    plt.title('Probability of each')
    plt.show()

def pie_chart(labels, values):
    plt.pie(values, labels=labels)
    plt.title("Pie Probability")
    plt.show()

# x = np.arange(-2.0, 6.0, 0.1)
# scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])
# plt.plot(x, softmax(scores).T, linewidth=2)

def np_array_to_strings(matrix):
    return np.array(matrix).astype(str)

labels = np_array_to_strings(scores)
print("labels",labels)
softmax_results=softmax(scores)




print("ndim", softmax_results.ndim)

if(softmax_results.ndim<2):
    bar_chart(labels, softmax_results.T)    
else:
    for label_row, results_row in zip(labels.T, softmax_results.T):
        print("labels: ",label_row)
        print("values: ",results_row )
        bar_chart(label_row, results_row)
        # pie_chart(label_row, results_row)