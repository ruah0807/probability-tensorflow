# conda activate tensorflow_macos
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 




import pandas as pd
# scv파일 읽는 명령어
data = pd.read_csv('gpascore.csv')


# 데이터 전처리  
#      : 혹시 비어있는 곳이 있다면 평균값을 넣거나, 지우거나 해야하기때문에 데이터 전처리가 필요함.
# print(data.isnull().sum())

data = data.dropna()    # dropna() : 빈칸 drop 함수
# data.fillna(100)      # fillna(?) : 빈칸채워주는함수

print(data['gpa'].min())    # min(): gpa 행의 최소값 확인 가능
print(data['gre'].max())    # max() : gre 행의 최대값
print(data['gre'].count())  # count() : 행의 갯수
exit()



import tensorflow as tf



# 딥러닝 모델
model = tf.keras.models.Sequencial([
    #  tf.keras.layers.Dense(노드의 갯수)
    #레이어에 activation Function 넣기 = 'tanh','relu','sigmore', etc.
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
    # 마지막 레이어는 항상 예측결과를 뱉어야함 'sigmoid' - 모든값을 0~1 사이로 압축시켜주는 activation Function.
]) 

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


### optimizer : 경사하강법으로 기울기를 뺄때 어떤 식으로 정할지 선택하는 것.
                # 상황에 맞게 바꿔야함. 
                # 종류 : adam, adagrad, adadelta, rmsprop, sgd, etc.
### loss(loss function): binary_crossentropy
                        # 결과가 0과 1 사이의 분류/ 확률 문제에서 씀.
                        
# epochs : 몇번 학습시킬것인가
model.fit( x, y, epochs =10 )

# require input for consequence = x = [수능성적, 학점, 대학교등급]
# x= [ [380,3.21,3], [660,3.67,3], [], []..]

# consequence : y = [합격여부]
# y = [ [0], [1], [0], [], [] ..]
