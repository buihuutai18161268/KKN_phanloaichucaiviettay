import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf 
import datetime as dt
# thư viện tách tập dữ liệu thành 2 cái khác nhau 1 dùng để train(huấn luyện), 1 dùng để test
from sklearn.model_selection import train_test_split
# đo độ chính xác của thuật toán 
from sklearn.metrics import accuracy_score
# datasets bao gồm các tập dữ liệu phổ biến như minist, iris flower
from sklearn import datasets,neighbors


print("Load MNIST Database")
mnist = tf.keras.datasets.mnist

# kích thước x_train là (60000,28,28) , x_test là (10000,28,28)
# x_train : data dữ liệu dùng để train(huấn luyện), y_train: label(tên hay nhãn) của x_train
# x_test : data dữ liệu dùng để test(kiểm tra), y_test: label(tên hay nhãn) của x_test
(x_train,y_train),(x_test,y_test) = mnist.load_data()

# định hình lại ma trận x_train có kích thước là (60000,784) và x_test có kích thước là (10000,784)
x_train = np.reshape(x_train,(60000,784))/255.0
x_test = np.reshape(x_test,(10000,784))/255.0

# tạo ra 1 model knn , k = 5 (lấy 5 điểm gần nhất với point)
knn =  neighbors.KNeighborsClassifier(n_neighbors=5)  
# huấn luyện lấy các điểm cho sẵn và lấy các label của nó
knn.fit(x_train,y_train)
# dự đoán label của các điểm dữ liệu X_test
y_predict = knn.predict(x_test)

# nhập vào data kiểm tra để dự đoán 
i = int(input("Nhập vào số thứ tự x_test để dự đoán:")) # vì input vào là kiểu dữ liệu String

# số dự đoán bằng thuật toán, reshape(1,-1) vd [1] [2] -> [[1] [2]] : cái mà thuật toán lấy
print("số dự đoán bằng thuật toán:",knn.predict(x_test[i].reshape(1, -1)))

# so sánh y_predict và y_test xem độ chính xác của thuật toán bao nhiêu
acc = accuracy_score(y_predict, y_test)
print("Độ chính xác thuật toán KNN:",acc*100,"%")

# ảnh trắng đen
plt.gray()
# X_test[0].reshape(28,28) chuyển từ (784,1) -> (28,28)
plt.imshow(x_test[i].reshape(28,28))
# hiển thị ảnh
plt.show()

