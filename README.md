# 480-project
## Install project files
```
git clone git@github.com:eyu776/480-project.git
```

## Installing Dependencies via Conda
```
conda create -n {name} python=3.8
conda activate {name}
pip install torch torchvision numpy scikit-learn pandas matplotlib
```

## To train
*Not required to run app, because there is already a saved pretrained model* `lstm.pth`.

To train, run:
```
python LSTM.py
```

`LSTM.py` trains the model on the training dataset in `data/enstars/enstars_data.csv` and saves the model in `lstm.pth`.

## To run
### Changing the input dataset
The model takes data in csv format with a single column per row. To set the dataset, modify lines 13 to 15 in `main.py`. (The default input dataset `enstars_data_test.csv` is already set.)

```
data_file_path = "./data/enstars/enstars_data_test.csv"
data_file_columns = ["banner"]
data_names_path = "./data/enstars/enstars_names_test.csv"
```

`data_file_path` takes in the filename for the file of the character gacha banners data as numbers.

`data_names_path` takes in the filename for the file of the character gacha banners data as strings.

`data_file_columns` takes in the header of the files in a string list.

To change to the second input dataset, modify to:

```
data_file_path = "./data/enstars/enstars_data_test2.csv"
data_file_columns = ["banner"]
data_names_path = "./data/enstars/enstars_names_test2.csv"
```

### Running the app
To run the app, run:
```
python main.py
```

`main.py` will run on the given dataset and predict the immediate next banner following the end of the dataset. The RMSE of the input dataset will be printed to the terminal. The predicted and actual gacha banners of the dataset will be displayed to the terminal, then the predicted immediate next banner will be printed to the terminal. A graph will also be displayed to the user, with the blue graph plotting the actual values and the red graph plotting the predicted values.