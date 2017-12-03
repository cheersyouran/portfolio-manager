I. Package Introduction:

    |-- State_Space: this package is used to generate daily state space and portfolio IR rank.
    |   |-- feat_selection.py: Functions mainly used to select features.
    |   |-- Generate_IR_rank_week.py: Generate information ratio rank based on weekly data.
    |   |-- Update_IR_rank.py: Functions used to update IR rank every week.
    |
    |-- data: this folder contains required data.
    |   |-- industry.csv : A dataset consists of industry information of all stocks.
    |   |-- industry_quote.xlsx : A dataset consists of the quote of all 29 industry indexes.
    |   |-- nav.csv : A dataset consists of portfolio net asset value information.
    |   |-- quote.csv : A dataset consists of market quotes of all stocks necessary for this project.
    |   |-- records.csv : A dataset that consists of portfolio rebalancing records on Xueqiu.
    |   |-- IR_rank_week.csv : A dataset consists of information ratio rank based on weekly performances of portfolios.
    |
    |-- task1: stock trading.
    |   |-- Similarity_Search : In this folder, we realized a Similarity Search method to allocate stock weights.
    |   |-- Supervised_Learning : In this folder, we realized a Supervised Learning method (Neural Network) to allocate stock weights.
    |
    |-- task2: portfolio management.
    |   |-- DDPGmodel: use DDPG model.
    |   |-- DQNmodel: use DQN model.
    |   |-- QLearningModel: use Q-learning model.
    |   |-- QLearningModel_beta: use Q-learning model and slide (train and test) window.
    |
    |-- base.py: some basic method used by all packages.
    |
    |-- run_task1.py: main function of running task1.
    |
    |-- run_task2.py: main function of running task2.


II. How to run the model?

    For each task, please modify PROJ_PATH at ./base.py
        PROJ_PATH='........./RL'

    Task1:
        1. Make sure to use Python 3.0+.
        2. Make sure to install all required third-part packages: [numpy, bitarray, pandas, time, etc].
        3. Replace data files in ./data folder, the file name MUST BE same as above introduction.
        4. Update 'project_path' and start_date in ./run_task1.py file.
              project_path='......../RL'
              start_date='2017-01-06' : that means our model will use data before 2017-01-05 to train a model, and make prediction on 2017-01-06.
        5. [OPTIONAL] You can change number of testing days, window size, number of stocks you choose from and output in task1/Similarity_Search/CONFIG.py file.
        6. python ./run_task1.py

        Then you can check the result file at "result/task1_result.csv"

    Task2:
        1. Make sure to use Python 3.0+.
        2. Make sure to install all required third-part packages: [numpy, Keras, tensorflow, TA-lib, pandas, etc].
        3. Replace data files in ./data folder, the file name MUST BE same as above introduction.
        4. Update 'project_path' and start_date in ./run_task2.py file.
              project_path='......../RL'
              start_date='2017-01-06' : that means our model will use data before 2017-01-05 to train a model, and make prediction on 2017-01-06.
        5. [OPTIONAL] You can change 'train_window' size in ./run_task2.py file.
              train_window = 10
        6. python ./run_task2.py

        Then you can check the result file at "result/task2_result.csv"

III. BUG?

    Of course...
    If some strange error appear, calm down please, re-run it or contact me thx! email: cheeryouran@yeah.net
    Have a good day!
