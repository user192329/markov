import pandas as pd
import numpy as np
import random as rm


filepath = 'C:\\Users\\Delyo\\Desktop\\BTCUSDT.csv'

data = pd.read_csv(filepath)
data = data.drop(columns=data.columns[0])
data.info()


# Price movment is defined by the difference in Open and Close prices

price_movement = data.open - data.close
price_movement.describe() #50% quantile is exactly at 0. The data is symetric.


# We classify the price movements in 3 categories: up, down, neutral.
# neutral movments will be 0 +- epsilon. How to choose epsilon?

def discretization(epsilon):
    bins = [min(price_movement)-1, 0-epsilon, 0+epsilon, max(price_movement)+1]
    labels = ['down', 'neutral', 'up']
    global move
    move = pd.cut(price_movement, bins = bins, labels = labels)
    return move

# We try to discretize the data by considering 
# changes in the interval (-std/2, std/2) to be neutral

discretization(np.std(price_movement)/2)
move.value_counts() 

# most of the time market is neutral. We will try to make the model 
# more sensitive to small changes by choosing epsilon = 2

discretization(2)
move.value_counts() 

# split move data in train and validation set 0.7/0.3
train_move = move[:int(len(move)*0.7)]
val_move = move[int(len(move)*0.7):]

# count the number of transitions between states:
transitions_count = {'UU':0, 'UN':0, 'UD':0, 
                     'NU':0, 'NN':0, 'ND':0,
                     'DU':0, 'DN':0, 'DD':0}    

# this fucntion could be improved
for i in range(1,len(train_move)): 
    if train_move[i-1] == 'up' and train_move[i] == 'up':
        transitions_count['UU'] += 1
    elif train_move[i-1] == 'up' and train_move[i] == 'neutral':
        transitions_count['UN'] += 1
    elif train_move[i-1] == 'up' and train_move[i] == 'down':
        transitions_count['UD'] += 1
    elif train_move[i-1] == 'neutral' and train_move[i] == 'up':
        transitions_count['NU'] += 1
    elif train_move[i-1] == 'neutral' and train_move[i] == 'neutral':
        transitions_count['NN'] += 1
    elif train_move[i-1] == 'neutral' and train_move[i] == 'down':
        transitions_count['ND'] += 1
    elif train_move[i-1] == 'down' and train_move[i] == 'up':
        transitions_count['DU'] += 1
    elif train_move[i-1] == 'down' and train_move[i] == 'neutral':
        transitions_count['DN'] += 1
    elif train_move[i-1] == 'down' and train_move[i] == 'down':
        transitions_count['DD'] += 1

print(transitions_count )       
transitions_arr = np.array(list(transitions_count.values()))

transitions_matrix = ([transitions_arr[:3]/sum(transitions_arr[:3]),
                       transitions_arr[3:6]/sum(transitions_arr[3:6]),
                       transitions_arr[6:9]/sum(transitions_arr[6:9])])

print(transitions_matrix)

# transition probabilities for up and down states are nearly equal
# it will be difficult to make predicitons

# stationary distribution
matrix_power(transitions_matrix, 100) 

# most of the time the market is in neutral state
# from neutral it can go to up or down state with equal probability

# Prediction

states = ['up','neutral','down']

def next_min():
    forecast = []
    stateToday = move.iloc[-1]
    if stateToday == 'up':
        forecast.append(np.random.choice(states, p=transitions_matrix[0]))
    elif stateToday == 'neutral':
        forecast.append(np.random.choice(states, p=transitions_matrix[1]))
    elif stateToday == 'down':
        forecast.append(np.random.choice(states, p=transitions_matrix[2]))
    
    print(forecast)
        
# forecast for the next closing price:
next_min()

def long_forecast(m):
    forecast = []
    stateToday = move.iloc[-1]
    i = 0
    while i != m:           
        if stateToday == 'up':
            forecast.append(np.random.choice(states, p=transitions_matrix[0]))
            stateToday = forecast[-1]
        elif stateToday == 'neutral':
            forecast.append(np.random.choice(states, p=transitions_matrix[1]))
            stateToday = forecast[-1]
        elif stateToday == 'down':
            forecast.append(np.random.choice(states, p=transitions_matrix[2]))
            stateToday = forecast[-1]
        i +=1
    return forecast
        
# forecast for the closing prices for the next 300 min
long_forecast(300)

# forecasting the whole validation data 

predicted = long_forecast(len(val_move))

accuracy = sum(predicted == val_move)/len(val_move)
accuracy # quite low

"""
The model could be further improved by adding more states
or taking in account trade volume or looking at shorter periods.
Having so many data points takes us close to the stationary distribution
very fast, whit equal probabilities for up and down changes.
Better approach would be looking at shorter periods in order to capture
price trends. For example 1h (60 observations) and trying to predict 
the next 1 min or 10.
"""

np.seterr(divide='ignore', invalid='ignore')


# function to get transition matrix for price movements in interval (a, b)
# and state at the end of interval

def counts(a,b):
    global current_state 
    current_state = train_move.iloc[b-1]
    transitions = {'UU':0, 'UN':0, 'UD':0, 
                   'NU':0, 'NN':0, 'ND':0,
                   'DU':0, 'DN':0, 'DD':0}
    
    for i in range(a,b): 
        if train_move[i-1] == 'up' and train_move[i] == 'up':
            transitions['UU'] += 1
        elif train_move[i-1] == 'up' and train_move[i] == 'neutral':
            transitions['UN'] += 1
        elif train_move[i-1] == 'up' and train_move[i] == 'down':
            transitions['UD'] += 1
        elif train_move[i-1] == 'neutral' and train_move[i] == 'up':
            transitions['NU'] += 1
        elif train_move[i-1] == 'neutral' and train_move[i] == 'neutral':
            transitions['NN'] += 1
        elif train_move[i-1] == 'neutral' and train_move[i] == 'down':
            transitions['ND'] += 1
        elif train_move[i-1] == 'down' and train_move[i] == 'up':
            transitions['DU'] += 1
        elif train_move[i-1] == 'down' and train_move[i] == 'neutral':
            transitions['DN'] += 1
        elif train_move[i-1] == 'down' and train_move[i] == 'down':
            transitions['DD'] += 1
        
        transitions_arr = np.array(list(transitions.values()))
        global p_matrix
        p_matrix = (transitions_arr[:3]/sum(transitions_arr[:3]), 
                    transitions_arr[3:6]/sum(transitions_arr[3:6]),
                    transitions_arr[6:9]/sum(transitions_arr[6:9]))
                           
    return p_matrix, current_state


# for the next minute move we choose the one with highest probability,
# based on the current state and transitions in the past hour

def next_min_move(p_matrix, current_state):
    global next_move
    next_possiblity = ['up', 'neutral', 'down']
    if current_state == 'up':
        next_move = next_possiblity[p_matrix[0].argmax()]
    elif current_state == 'neutral':
        next_move = next_possiblity[p_matrix[1].argmax()]
    elif current_state == 'down':
        next_move = next_possiblity[p_matrix[2].argmax()]
    
    return next_move


# simulating 10000 1h samples and the prediction for next state. Might take time
ls = []
for i in range(1,1000):
    ls.append(next_min_move(counts(i, i+60)[0], counts(i, i+60)[1]))

sum(train_move[61:1060] == ls)/len(ls) #accuracy


# Maybe we should only enter the market when the probability
# for entering a certain state is over 50%?

def next_min_move_skips(p_matrix, current_state):
    global next_move
    next_possiblity = ['up', 'neutral', 'down']
    if current_state == 'up' and max(p_matrix[0]) > 0.5:
        next_move = next_possiblity[p_matrix[0].argmax()]        
    elif current_state == 'neutral' and max(p_matrix[1]) > 0.5:
        next_move = next_possiblity[p_matrix[1].argmax()]
    elif current_state == 'down' and max(p_matrix[2]) > 0.5:
        next_move = next_possiblity[p_matrix[2].argmax()]
    else:
        next_move = 'skip'
    
    return next_move



next_min_move_skips(p_matrix, current_state)

ls1 = []
for i in range(1,1000):
    ls1.append(next_min_move_skips(counts(i, i+60)[0], counts(i, i+60)[1]))

# correct predictions over total number of taken actions (non-skips)    
sum(ls1 == train_move[61:1060])/sum(pd.Series(ls1) != 'skip')


# Other ways of improving the model should be considered.


