# Tiny-ML IR-Sensor

The source code for building a machine learning model is presented below. The model is created in Python using [Colab](https://colab.research.google.com). Before you can start using the source code, you need to get a free google account. Follow the introduction video linked below to get a first impression of Colab (link the video...). If you need further information check out the following [link](https://towardsdatascience.com/getting-started-with-google-colab-f2fff97f594c).
You can also train the model in an equivalent local environment. 

## Workflow 
The picture below illustrates the process of the model development. After the data is generated with the sensor, the data will be written to a CSV file via Node-RED. The label is also added to the dataset via Node-RED. In the next step the CSV file will be uploaded to Colab and the model is trained. After training, the model will be converted into a TinyML. The Tiny ML can be used to classify new sensor data via Colab. In the following the steps are described in detail. 

![alt-text][Workflow]
[Workflow]: 

## Data Preparation 

1) Load the data 

    In the very first step we upload the given csv file to google drive. The csv file is called dataset.csv. Instead of using google drive you can also upload the data directly to colab. However, when using colab you need to upload the data every single time you start using colab. The code below uses google drive. Please make sure to upload the csv file directly to google drive. Do not choose an existing folder. Follow the introduction video (erneut das Video verlinken..) for how to upload a file to colab.
    The following code loads the data into colab and initializes it to a dataframe. 
    
    ```python
    url = "/content/drive/MyDrive/dataset.csv"

    names = ["Date", "data0", "data1", "data2", "data3", "data4", "data5", "data6", \
         "data7", "data8", "data9", "data10", "data11", "data12", "data13", "data14", \
         "data15", "data16", "data17", "data18", "data19", "data20", "data21", "data22", \
         "data23", "data24", "data25", "data26", "data27", "data28", "data29", "data30", \
         "data31", "data32", "data33", "data34", "data35", "data36", "data37", "data38", \
         "data39", "data40", "data41", "data42", "data43", "data44", "data45", "data46",\
         "data47", "data48", "data49", "data50", "data51", "data52", "data53", "data54", \
         "data55", "data56", "data57", "data58", "data59", "data60", "data61", "data62", \
         "data63", "label"]

    df = pd.read_csv(url, header=None, names=names, parse_dates=True, sep=',')    
    ```
    
    Instead of using the regular index we set the timestamp to the index.  
    ```python 
    df["Date"] = pd.to_datetime(df["Date"]) 
    df.index = df["Date"]
    del df["Date"]
    ```

    After the first cell is executed, you expect the following output. If you receive a warning or error message, check if the file was added to the correct folder. 
    
    ![alt-text][Output_data]

    [Output_data]: https://github.com/SSV-embedded/TinyML_IR-Sensor/blob/452beb9e039a98f3a2c42915b45f6c3f3183a839/Output_data.png

2) Seperate the labels 

    Moreover we separate the labels from the dataset. Df holds the temperature values and df_label holds 
    the labels. Both dataframes contain the timestamps. 

    ```python 
    df_label =  pd.concat([df.pop(x) for x in ['label']], axis = 1)
    ```

3) Get to know the data 

    The dataset was created with an IR-Temperature sensor which generates an 8x8 array with temperature values. All values are recorded in degrees celsius. 
    
    Class 0 describes the state ok. Class 1 describes a critical state.

    
    ![alt-text][Bild]
    
    [Bild]: https://github.com/SSV-embedded/TinyML_IR-Sensor/blob/2cf61b172882a23991e954e01e803f0b2f337289/Bild_Github.png
 
4) Normalite the Data

    Before the data can be split into a training and testing dataset, the data needs to be normalized. Therefor, the maximum and the minimum of the entire data frame must be determined.
    
    ```python 
    min = df.min().min()
    max = df.max().max()
    ``` 
    In order to normalize the dataframe we use the following equation: (x - min ) / (max - min).
    
    ```python 
    df_norm = (df - min) / (max - min) 
    ```

    Choosing a Min-Max-Normalization, the data will be transformed into a linear range. Thus, the minimum value is scaled to 0 and the maximum value is scaled to 1. In addition to Min-Max-Normalization, Z-value normalization 
    and decimal scaling also exist. Normalizing the data reduces the likelihood of failures and therefore optimizes the model.

5) Split Data into Subsets 


    Following the data is split into a training and testing dataset. The training dataset is used to train the machine learning model. The testing dataset evaluates the trained machine learning model. We set random_state=0 to enable reproducibility. This is necessary, because the temperature values and the labels are already in two different datasets. Random_state=0 makes sure that index of train_dataset and test_dataset fit the index order of train_labels and test_labels. 80% of the data belong to the train_dataset. 20% of the data belong to the test_dataset. 
    
    ```python 
    #Set Dataset 
    train_dataset = df_norm.sample(frac=0.8, random_state=0)
    test_dataset = df_norm.drop(train_dataset.index)

    #Set Labels 
    train_labels = df_label.sample(frac=0.8, random_state=0)
    test_labels = df_label.drop(train_dataset.index)
    ```

## Model Training 
1) Import Tensorflow framework

    ```python 
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.utils import to_categorical
    ```

    
2) Build the model 

    To classify the temperature images, we use a rather simple Artifical Neural Network (ANN) consisting of only one layer. We set the layer type to Dense. This means that each neuron in a layer receives input from every neuron from the previous layer. Thus the Dense layer is fully connected. For binary classification we use a sigmoid function as an activation function. A sigmoid function is often used as the last layer of a machine learning model because the function converts the model's output into a probability score. As we only use one layer, we directly use the sigmoid function. For multiple classification one can use a softmax function as the last layer instead. 

    ``` python 
    model = tf.keras.Sequential([
      layers.Dense(num_classes, activation = 'sigmoid', input_shape=(input,)),
        ])
    model.summary()
    ```

    Furthermore we need to set the optimizer, the learning rate and the loss function. In this example the Adam optimizer was selected. This optimizer updates the parameters in such way that it speeds up reaching the minima of the cost function. The learning rate determines the step size at each iteration while moving towards the minimum of the cost function. We set the learning rate to 0.005. The loss function computes the loss between actual and predicted class labels. We use a categorical crossentropy loss function. 

    ```python 
    from tensorflow.keras.optimizers import Adam

    model.compile(
    optimizer=Adam(lr= 0.005), 
    loss = 'categorical_crossentropy', 
    metrics = ['accuracy'],
    )
    ```


