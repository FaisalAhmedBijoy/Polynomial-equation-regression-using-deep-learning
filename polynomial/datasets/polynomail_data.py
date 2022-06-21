
from traceback import print_tb
import matplotlib.pyplot as plt

def polynomial_equation(x):
    return x**2
def create_dataset(range_number):
    dataset = []
    x=[i for i in range(range_number)]
    y=[polynomial_equation(i) for i in x]
    return x,y 

if __name__ == '__main__':
    x,y=create_dataset(range_number=100)
   
    print(x)
    print(y)
    plt.scatter(x,y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('x vs y')
    plt.show()
   