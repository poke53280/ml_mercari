
from keras.wrappers.scikit_learn import KerasClassifier



local_dir = os.getenv('LOCAL_PY_DIR')

assert local_dir is not None, "Set environment variable LOCAL_PY_DIR to parent folder of ai_lab_datapipe folder. Instructions in code."

# * 'Windows/Start'button
# * Type 'environment...'. 
# * Select option 'Edit environment variables for your account' (NOT: 'Edit the system environment variables)
# * New. LOCAL_PY_DIR. Value is parent folder to ai_lab_datapipe
# * Restart IDE/python context. 

print(f"Local python top directoy is set to {local_dir}")
os.chdir(local_dir)

DATA_DIR_PORTABLE = "C:\\plasticc_data\\"
DATA_DIR_BASEMENT = "D:\\XXX\\"
DATA_DIR = DATA_DIR_PORTABLE

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)

df = pd.read_pickle(DATA_DIR + "df_t_1000_wide_2018-10-22_11-25.pkl")

y = df['target_90']
df = df.drop(['target_90'], axis = 1)

y = np.array(y, dtype = float)
y = y.astype(np.float32)

data = df.values
data = data.astype(np.float32)


# Fully connected vs. rnn, 1d conv

# https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/

# baseline model
def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=15, shuffle=True, random_state=seed)
results = cross_val_score(estimator, data, y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))




