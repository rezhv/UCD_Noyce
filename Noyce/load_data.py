
import pandas as pd
from utils.normalizer import normalize
from sklearn.model_selection import train_test_split

def load_pol_data():

    df = pd.read_csv(
        "./UCD_Noyce/Noyce/data/political_text/political_train.csv", encoding='unicode_escape')
    df_test = pd.read_csv(
        "./UCD_Noyce/Noyce/data/political_text/political_test.csv", encoding='unicode_escape')
    df['text'] = df['text'].apply(normalize)
    df_test['text'] = df_test['text'].apply(normalize)
    return df['text'].tolist(), df['class_id'].tolist(), df_test['text'].tolist(), df_test['class_id'].tolist()

def load_ideology_data():

    df = pd.read_csv(
        "./UCD_Noyce/Noyce/data/ideology/facebook.csv", encoding='unicode_escape')

    df['text'] = df['text'].apply(normalize)
    df = df.dropna()

    df ,df_test = train_test_split(df, random_state=1, test_size=0.1, stratify = df['class_id'])

    return df['text'].tolist(), df['class_id'].tolist(), df_test['text'].tolist(), df_test['class_id'].tolist()

def load_disagreement_data():
    class_id_dict = {
        "SE" : 0,
        "AC" : 0,
        "AE" : 0,
        "DE" : 1,
        "DC" : 1,
        "DC/AC" : 1,
        "DE/DC" : 1,

    }

    YT_df = pd.read_csv(
        "./UCD_Noyce/Noyce/data/disagreement/Youtube_Disagreement_Comments.csv", encoding='unicode_escape')[['text','class']]

    FB_df = pd.read_csv(
        "./UCD_Noyce/Noyce/data/disagreement/Facebook_Disagreement_Comments.csv", encoding='unicode_escape')[['text','class']]

    Reddit_df = pd.read_csv(
        "./UCD_Noyce/Noyce/data/disagreement/Reddit_Disagreement_Comments.csv", encoding='unicode_escape')[['text','class']]

    df = pd.concat([YT_df, FB_df, Reddit_df])
    df['class_id'] = df['class'].map(class_id_dict)
    df = df.dropna()

    
    df_train ,df_test = train_test_split(df, random_state=1, test_size=0.1, stratify = df['class_id'])

    
    df_train.loc[:,'text'] = df_train.text.apply(normalize)
    df_test.loc[:,'text'] = df_test.text.apply(normalize)

    return df_train['text'].tolist(), df_train['class_id'].tolist(), df_test['text'].tolist(), df_test['class_id'].tolist()


def load_data(dset_name='political_text'):
    if dset_name == 'political_text':
        return load_pol_data()
    elif dset_name == 'disagreement':
        return load_disagreement_data()
    elif dset_name == 'ideology':
        return load_ideology_data()
    else:
        raise NameError(
            'Dataset not known. Available Datasets: political_text')

if __name__ == '__main__':
    print(load_data('ideology')[2])