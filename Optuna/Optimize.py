# Test database access
import mysql.connector
cnx = mysql.connector.connect(user='cody', password='dkmbdirt',
                              host='192.168.4.24',
                              database='lightfm')
print(cnx)
# List all tables in the database
cursor = cnx.cursor()
cursor.execute("SHOW TABLES")
for (table_name,) in cursor:
    print(table_name)
cnx.close()

import pandas as pd

from lightfm import LightFM
from lightfm.evaluation import precision_at_k,auc_score,reciprocal_rank
from lightfm.data import Dataset
from lightfm import LightFM, cross_validation
import numpy as np

# Import data
philly_bus = pd.read_feather('FilteredData/business_philly.feather')
philly_reviews = pd.read_feather('FilteredData/review_philly.feather')
philly_users = pd.read_feather('FilteredData/user_philly.feather')

df = philly_reviews.groupby('user_id')['stars'].mean()
users = pd.merge(philly_users, df, on=['user_id'], how='left')

bins = [0, 0.9999999, 1.9999999, 2.9999999, 3.9999999, 4.9999999, 5]
labels = ["0","1", "2", "3","4", "5"]
users["star_bin"] = pd.cut(users['stars'], bins=bins, labels=labels)

reviews_only = philly_reviews[["user_id", "business_id", "stars"]]

#unique user features
user_f = []
user_col = ['star_bin']*len(users['star_bin'].unique()) 
user_unique_list = list(users['star_bin'].unique())
# col = ['review_count']*len(users['review_count'].unique()) + ['useful']*len(users['useful'].unique()) + ['funny']*len(users['funny'].unique()) + ['cool']*len(users['cool'].unique())
# unique_list = list(users['review_count'].unique()) + list(users['useful'].unique()) + list(users['funny'].unique()) + list(users['cool'].unique())


for x,y in zip(user_col, user_unique_list):
    res = str(x)+ ":" +str(y)
    user_f.append(res)

#unique item features
item_f = []
item_col = ['stars']*len(philly_bus['stars'].unique()) + ['postal_code']*len(philly_bus['postal_code'].unique())
item_unique_list = list(philly_bus['stars'].unique()) + list(philly_bus['postal_code'].unique())

for x,y in zip(item_col, item_unique_list):
    res = str(x)+ ":" +str(y)
    item_f.append(res)
#     print(res)

dataset1 = Dataset()
dataset1.fit(
        philly_reviews['user_id'].unique(), # all the users
        philly_reviews['business_id'].unique(), # all the items
        user_features = user_f, # additional user features
        item_features = item_f #additional item features
)

(interactions, weights) = dataset1.build_interactions([(x[0], x[1], x[2]) for x in reviews_only.values])

train, test = cross_validation.random_train_test_split(interactions, test_percentage=0.25, random_state=np.random.RandomState(42))

# Split train into train and validation
train, validation = cross_validation.random_train_test_split(train, test_percentage=0.25, random_state=np.random.RandomState(42))

print('Train set size: ', train.shape)
print('Validation set size: ', validation.shape)

import optuna
import lightfm
# Define search space
def objective(trial):
    # Build a list of hyperparameters to pass to the model
    hyperparameters = {
        'no_components': trial.suggest_int('no_components', 10, 100),
        'learning_schedule': trial.suggest_categorical('learning_schedule', ['adagrad', 'adadelta']),
        'loss': trial.suggest_categorical('loss', ['warp', 'bpr', 'warp-kos','logistic']),
        'user_alpha': trial.suggest_float('user_alpha', 1e-5, 1e-2),
    }

    # Create WARP model
    if hyperparameters['loss']=='warp':
        # Add warp specific parameters
        "empty"
        
    elif hyperparameters['loss']=='bpr':
        "empty"
    elif hyperparameters['loss']=='warp-kos':
        # Add warp-kos specific parameters
        hyperparameters['k'] = trial.suggest_int('k', 1, 10)
        hyperparameters['n'] = trial.suggest_int('n', 1, 10)
    elif hyperparameters['loss']=='logistic':
        "empty"
    else:
        print('Error: loss function not found')
        return
    
    if hyperparameters['learning_schedule']=='adagrad':
        # Add adagrad specific parameters
        hyperparameters['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    
    elif hyperparameters['learning_schedule']=='adadelta':
        hyperparameters['rho'] = trial.suggest_float('rho ', 0.1, 0.9)
        hyperparameters['epsilon'] = trial.suggest_float('epsilon', 1e-5, 1e-1, log=True)
        hyperparameters['item_alpha'] = trial.suggest_float('item_alpha ', 1e-5, 1e-1, log=True)
    
    # Set up lightfm model with data
    model=lightfm.LightFM(**hyperparameters)

    model.fit(train, epochs=10, num_threads=2)

    # Get AUC score
    auc = auc_score(model, validation, train_interactions=train, num_threads=2, check_intersections=False).mean()

    return auc
# Run the optimization
study2 = optuna.create_study(
    study_name='lightFM Tuning 2',
    # Databsae is mysql on 192.168.4.24:3306
    # Database name is lightfm
    storage='mysql://cody:dkmbdirt@192.168.4.24:3306/lightfm',
    load_if_exists=True,
    direction='maximize',
)

# Run the optimization
study2 = optuna.create_study(
    study_name='lightFM Tuning 2',
    # Databsae is mysql on 192.168.4.24:3306
    # Database name is lightfm
    storage='mysql://cody:dkmbdirt@192.168.4.24:3306/lightfm',
    load_if_exists=True,
    direction='maximize',
)

# Run the optimization
study2.optimize(objective, n_trials=100)