# conda activate tensorflow_macos
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 




import pandas as pd


# scv파일 읽는 명령어
data = pd.read_csv('gpascore.csv')

# dropna() : 빈칸 drop 함수
data = data.dropna()   

# admit 열에 있는 data 담기
y_data = data['admit'].values

x_data = []


# x_data = [] <- 배열안에 정보 넣기
                # iterrow() : datafram(행)을 한줄씩 출력 가능 (ex) print(rows['gre'])
for i, rows in data.iterrows() :     
    # .append() : x_data 안에 데이터를 집어넣어주는 함수
    x_data.append([ rows['gre'], rows['gpa'], rows['rank'] ]) 


## numpy : 리스트 안에 리스트를 만들 때
import numpy as np
import tensorflow as tf



# 딥러닝 모델
model = tf.keras.models.Sequential([
    #  tf.keras.layers.Dense(노드의 갯수)
    #레이어에 activation Function 넣기 = 'tanh','relu','sigmore', etc.
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
    # 마지막 레이어는 항상 예측결과를 뱉어야함 'sigmoid' - 모든값을 0~1 사이로 압축시켜주는 activation Function.
]) 

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



model.fit( np.array(x_data), np.array(y_data), epochs =1000 ) 

# 예측

input_data = np.array([ [750, 3.70, 3],[400, 2.2, 1] ] )
예측값 = model.predict(input_data)
print(예측값)