import os

os.environ["CLEARML_WEB_HOST"] = "https://app.clear.ml"
os.environ["CLEARML_API_HOST"] = "https://api.clear.ml"
os.environ["CLEARML_FILES_HOST"] = "https://files.clear.ml"
os.environ["CLEARML_API_ACCESS_KEY"] = "18HHWGEGNFE1NP7YUZCC28YBMXY4WK"
os.environ["CLEARML_API_SECRET_KEY"] = "5r6wD71a9Rs4q5i1lOUdYVkp6Cn47QcsVQ7sMv4vNHXqCkhvtWO_5CY7kRz0cJDr-QE"

from clearml import PipelineController


# p1 - download - titanic
def download_data(pickle_data_url):
    import pandas as pd
    url = 'https://raw.githubusercontent.com/VoroninMaxim/Project_ML/main/diabetes.csv'
    data_frame = pd.read_csv(url)
    print("Data downloaded successfully. Shape:", data_frame.shape)
    return data_frame


# p2 - preprocessing
def preprocess_data(data_frame, test_size=0.2, random_stage=42):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # df_preproc = data_frame.drop(columns=['name', 'sex', 'cabin', 'embarked', 'boat', 'body', 'home.dest', 'ticket'])
    df_preproc = data_frame.drop(columns=['Insulin', 'DiabetesPedigreeFunction'])

    print("DataFrame after dropping columns shape:", df_preproc.shape)

    # Удаление строк с пропущенными значениями
    df_preproc = df_preproc.dropna()
    print("DataFrame after dropping rows with missing values shape:", df_preproc.shape)

    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Age', 'Outcome']:
        df_preproc[col] = df_preproc[col].fillna(df_preproc[col].median()).astype(int)

    train, test = train_test_split(df_preproc, test_size=test_size, random_state=random_stage)
    print("Train DataFrame shape:", train.shape)
    print("Test DataFrame shape:", test.shape)

    X_train = train.drop(columns=['Outcome'])
    y_train = train['Outcome']

    X_test = test.drop(columns=['Outcome'])
    y_test = test['Outcome']

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    return X_train, X_test, y_train, y_test


from sklearn.metrics import confusion_matrix, roc_curve, auc


# p3.1 - train model
def train_model(data):
    import pandas as pd
    from lightgbm import LGBMClassifier
    from clearml import Logger

    # Инциируем объект логирования
    # log = Logger.current_logger()
    # Проверяем, запущен ли код в ClearML
    if Logger.current_logger() is not None:
        # Инциализируем объект логирования
        log = Logger.current_logger()
    else:
        log = None

    # Проверяем, что data является кортежем с четырьмя элементами
    assert isinstance(data, tuple) and len(data) == 4, "Expected data to be a tuple with four elements"

    # Проверяем, что каждый элемент data является DataFrame или Series из библиотеки pandas
    assert all(isinstance(item, (pd.DataFrame, pd.Series)) for item in
               data), "Expected data to contain only DataFrame or Series objects"

    X_train, X_test, y_train, y_test = data

    model = LGBMClassifier(silent=True)
    model.fit(X_train, y_train, eval_metric=['binary_logloss', 'auc'], eval_set=[(X_train, y_train), (X_test, y_test)])

    # Оцениваем модель
    test_scores = model.evals_result_['valid_1']
    test_logloss = test_scores['binary_logloss'][-1]
    test_roc_auc = test_scores['auc'][-1]

    # Оцениваем модель
    train_scores = model.evals_result_['training']
    train_logloss = train_scores['binary_logloss'][-1]
    train_roc_auc = train_scores['auc'][-1]

    # Записываем метрики в ClearML

    if log is not None:
        log.report_scalar("Logloss", "Test", iteration=1, value=test_logloss)
        log.report_scalar("Logloss", "Train", iteration=1, value=train_logloss)

        log.report_scalar("ROC AUC", "Test", iteration=1, value=test_roc_auc)
        log.report_scalar("ROC AUC", "Train", iteration=1, value=train_roc_auc)

    else:
        # Выводим метрики в консоль
        print("Test Logloss:", test_logloss)
        print("Train Logloss:", train_logloss)
        print("Test ROC AUC:", test_roc_auc)
        print("Train ROC AUC:", train_roc_auc)

    return model


# p3.2 - parameters optimizer
def params_optimizer(data):
    import pandas as pd
    from lightgbm import LGBMClassifier
    from clearml import Logger, Task
    from sklearn.model_selection import train_test_split, ParameterSampler
    import joblib

    # Инциируем объект логирования
    # log = Logger.current_logger()
    # Проверяем, запущен ли код в ClearML
    if Logger.current_logger() is not None:
        # Инциализируем объект логирования
        log = Logger.current_logger()
    else:
        log = None

    X_train, X_test, y_train, y_test = data

    param_grid = {
        'depth': [4, 5, 6, 7, 8],  # глубина на которой пойдет lightgbm
        'learning_rate': [0.1, 0.05, 0.01, 0.005, 0.001],  # используется грязный поиск ОПТУНА более умнее
        'iterations': [30, 50, 100, 150]
    }

    # Переменные для хранения результатов
    best_score = 0
    best_model = None
    i = 0

    # for param in ParameterSampler(param_grid, n_iter=20, random_state=42):
    for i, param in enumerate(ParameterSampler(param_grid, n_iter=20, random_state=42)):
        print(param)

        # Проверяем, запущен ли код в ClearML
        if Task.current_task() is not None:
            # paramas_dict = Task.current_task().connect(parameters=param)
            # Task.current_task().connect(parameters=param)
            Task.current_task().set_parameters(param)

        print(i + 1)

        # Обучаем модель
        model = LGBMClassifier(**param, silent=True)
        model.fit(X_train, y_train, eval_metric=['binary_logloss', 'auc'],
                  eval_set=[(X_train, y_train), (X_test, y_test)])

        # Оцениваем модель
        test_logloss, test_roc_auc = model.evals_result_['valid_1']['binary_logloss'][-1], \
        model.evals_result_['valid_1']['auc'][-1]
        train_logloss, train_roc_auc = model.evals_result_['training']['binary_logloss'][-1], \
        model.evals_result_['training']['auc'][-1]

        # Сравниваем текущий скор с лучшим
        if test_roc_auc > best_score:
            # Save the model
            best_score = test_roc_auc
            best_model = model

            if log is not None:
                # Записываем метрики в ClearML
                log.report_scalar("Logloss", "Test", iteration=i, value=test_logloss)
                log.report_scalar("Logloss", "Train", iteration=i, value=train_logloss)

                log.report_scalar("ROC AUC", "Test", iteration=i, value=test_roc_auc)
                log.report_scalar("ROC AUC", "Train", iteration=i, value=train_roc_auc)

            # Сохраняем лучшую модель в ClearML
            if Task.current_task() is not None:
                joblib.dump(best_model, 'best_model.pkl')
                Task.current_task().upload_artifact('best_model', 'best_model.pkl')

            i += 1

    return best_model


# ----------------------------------------------------------------------------------------------------------------

pipe = PipelineController(project='ClearML_Webinar_2',
                          name='Webinar_pipeline',
                          version='1.1',
                          add_pipeline_tags=False)

pipe.add_parameter(
    name='url',
    description='url for dataset',
    default='https://raw.githubusercontent.com/VoroninMaxim/Project_ML/main/diabetes.csv'
)

pipe.add_function_step(
    name='download_data',
    function=download_data,
    function_return=['data_frame'],
    cache_executed_step=False
)

pipe.add_function_step(
    name='preprocess_data',
    parents=['download_data'],  # после какого этапа запускается наша функция
    function=preprocess_data,
    function_kwargs=dict(data_frame='${download_data.data_frame}'),
    function_return=['preprocess_data'],
    cache_executed_step=False
)

pipe.add_function_step(
    name='train_model',
    parents=['preprocess_data'],  # после какого этапа запускается наша функция
    function=train_model,
    function_kwargs=dict(data='${preprocess_data.preprocess_data}'),
    function_return=['model'],
    cache_executed_step=False
)

# - тут будет разветвтление на 2 части
pipe.add_function_step(
    name='params_optimizer',
    parents=['preprocess_data'],  # после какого этапа запускается наша функция
    function=params_optimizer,
    function_kwargs=dict(data='${preprocess_data.preprocess_data}'),
    function_return=['best_model'],
    cache_executed_step=False
)

pipe.start_locally(run_pipeline_steps_locally=True)
print('Pipline finished')