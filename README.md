# Tiny-ML IR-Sensor

The source code for building a machine learning model is presented below. The model is created in Python using [Colab](https://colab.research.google.com). Before you can start using the source code, you need to get a free google account. Follow the introduction video linked below to get a first impression of Colab (link the video...) If you need further information check out the following [link](https://towardsdatascience.com/getting-started-with-google-colab-f2fff97f594c).
You can also train the model in an equivalent local environment. 

## Data Preparation 

1) Load the data 

    In the very first step we upload the given csv file to google drive. The csv file is called dataset.csv. Instead of using google drive you can also upload the data directly to colab. However, when using colab you need to upload the data every single time you start using colab. The code below uses google drive. Please make sure to upload the csv file directly to google drive. Do not choose an existing folder. Follow the introduction video (erneut das Video verlinken...) for how to upload a file to colab.
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
    Moreover we separate the labels from the dataset. Df holds the temperature values and df_label holds 
    the labels. Both dataframes contain the timestamps. 
    ```python 
    df_label =  pd.concat([df.pop(x) for x in ['label']], axis = 1)
    ```

    After the first cell is executed, you expect the following output. If you receive a warning or error message, check if the file was added to the correct folder. 
    
    ![alt-text][Output_data]

    [Output_data]: https://github.com/SSV-embedded/TinyML_IR-Sensor/blob/d2691e7bbba615b89f12176a07e70cc2fe7cc7a3/Output_data.png


2) Get to know the data 

    The dataset was created with an IR-Temperature sensor which generates an 8x8 array with temperature values. All values are recorded in degrees celsius. 
    
    Class 0 describes the state ok. Class 1 describes a critical state.

    
    ![alt-text][Bild]
    
    [Bild]: https://github.com/SSV-embedded/TinyML_IR-Sensor/blob/2cf61b172882a23991e954e01e803f0b2f337289/Bild_Github.png
 
