import pandas as pd
import re
from sklearn.model_selection import train_test_split

def convert(file_path,filename):
    """Takes a file path and filename as inputs and returns a pandas dataframe. Will automatically detect what file type it is (.csv, .tsv or .parquet)"""
    file = f"{file_path}/{filename}"
    if file.endswith('.csv'):
        data = pd.read_csv(file)
        #return data
    elif file.endswith('.tsv'):
        data = pd.read_csv(file, sep = '\t')
        #return data
    elif file.endswith('.parquet'):
        data = pd.read_parquet(file)
    #elif file.endswith('.xls','xlsx'):
        #data = pd.read_excel(file)
        #return(data)
    else:
        raise ValueError(f'Unsupported filetype: {file}')
    return data

def splitter(X,y):

    """Takes text (X) and label (y) as inputs and returns test, train and validation data: X_test, y_test, X_train, y_train, X_val, y_val"""

    #Split off test data
    X_train_val, X_test, y_train_val, y_test =\
        train_test_split(X, y, test_size=0.1, random_state=2020)

    #Split remaining data into train and validation data
    X_train, X_val, y_train, y_val =\
        train_test_split(X_train_val, y_train_val, test_size=0.1111, random_state=2020)
    
    return X_test, y_test, X_train, y_train, X_val, y_val

# Function for cleaning text
def text_preprocessing(text): 
    """
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    #Remove line breaks
    text = re.sub(r'<br /><br />','',text).strip()

    return text

def sized_export(df, size):
    def get_var_name(var):                          #function for getting the name of a variable in string format
        for name, value in globals().items():
            if value is var:
                return name
    clean_root = "C:/Users/Bijan-PC/Documents/Coding/UNIL/Data Analysis/ADA_Project/ADA_Final/dat_cleaned"      #base location for where all cleaned data files are saved

    df_name = get_var_name(df)

    rows = df.groupby('label',group_keys=False)

    new_df = pd.DataFrame(rows.apply(lambda x: x.sample(size, random_state = 2020).reset_index(drop=True)))
    
    return new_df.to_csv(f"{clean_root}/{size}" + f'/{df_name}_{size}.csv', index=False)  

#def balance(df):
#    g = df.groupby('label')
#    x.sample(g.size().min()).reset_index(drop=True)

#original:
#g = df.groupby('label')
#g = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
    

if __name__ == '__main__':
    # here or wherever it is used
    file_path = input("enter the path to the file you want to open")
    df = convert(file_path)