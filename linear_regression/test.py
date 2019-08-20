import numpy as np
import sys
from LinearRegression import LinearRegression as LR


df = np.genfromtxt(sys.argv[1], delimiter=',')
raw_data = df[1:, 3:].copy()
where_is_nan = np.isnan(raw_data)
raw_data[where_is_nan] = 0

month_data = {}

for month in range(12):
    sample = np.zeros(shape=(18, 20*24))
    for day in range(20):
        for hour in range(24):
            row_start = 18 * (month * (20) + day)
            row_end = 18 * (month * (20) + day + 1)
            sample[:, day * 24 + hour] = raw_data[row_start: row_end, hour]
    month_data[month] = sample

x_train = np.zeros(shape=(12 * 471, 18 * 9))
y_train = np.zeros(shape=(12 * 471, 1))

for month in range(12):
    for hour in range(471):
        x_train[hour + month * 471, :] = month_data[month][:, hour:hour+9].reshape(1, -1)
        y_train[hour + month * 471, 0] = month_data[month][9, hour + 9]


x_test_data = np.array([[1,3,4],[4,5,3],[4,1,4]])
y_test_data = np.array([1,2,3])

model = LR()
model.load_data(x_train, y_train)
model.z_score_normal()
w_grad, b_grad = model.train(optimizer='Adam',lr=0.2 , batch_size=int(len(x_train)/1),  epoch = 2000)
w_ofs, b_ofs = model.closed_form_solution()

for i in range(len(w_grad)):
    print(w_grad[i],w_ofs[i])
print(b_grad, ' ',b_ofs)
