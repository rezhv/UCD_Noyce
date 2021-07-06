
import pandas as pd
from utils.normalizer import normalize
from sklearn.model_selection import train_test_split

FACEBOOK_POSTS = "./UCD_Noyce/Noyce/data/ideology/facebook.csv"
YOUTUBE_POSTS = "./UCD_Noyce/Noyce/data/ideology/youtube.csv"
REDDIT_COMMENTS = "./UCD_Noyce/Noyce/data/ideology/reddit_comments_onesided.csv"
REDDIT_COMMENTS_POL_BALANCED = "./UCD_Noyce/Noyce/data/ideology/reddit_comments_80_confidence.csv"

def load_pol_data():

    df = pd.read_csv(
        "./UCD_Noyce/Noyce/data/political_text/political_train.csv", encoding='unicode_escape')
    df_test = pd.read_csv(
        "./UCD_Noyce/Noyce/data/political_text/political_test.csv", encoding='unicode_escape')
    df['text'] = df['text'].apply(normalize)
    df_test['text'] = df_test['text'].apply(normalize)
    return df['text'].tolist(), df['class_id'].tolist(), df_test['text'].tolist(), df_test['class_id'].tolist()

def load_csv(path):
    df = pd.read_csv(path, encoding='unicode_escape')
    df['text'] = df['text'].apply(normalize)
    return df['text'].tolist()

def load_ideology_data(website, separate_websites = False, test_set = True):
    test_size = 0.1
    if (website == 'facebook'):
        path = FACEBOOK_POSTS
        test_size = 0.025
        df = pd.read_csv(path, encoding='unicode_escape')

    elif (website == 'youtube'):
        path = YOUTUBE_POSTS
        df = pd.read_csv(path, encoding='unicode_escape')

    elif (website == 'redditcomments'):
        path = REDDIT_COMMENTS
        df = pd.read_csv(path, encoding='unicode_escape')
        test_size = 0.01


    elif (website == 'youtube_facebook'):
        df1 = pd.read_csv(YOUTUBE_POSTS, encoding='unicode_escape')
        df2 = pd.read_csv(FACEBOOK_POSTS, encoding='unicode_escape')
        df = pd.concat([df1,df2])
        test_size = 0.020

    elif (website == 'all'):
        df1 = pd.read_csv(YOUTUBE_POSTS, encoding='unicode_escape')
        df2 = pd.read_csv(FACEBOOK_POSTS, encoding='unicode_escape')
        df3 = pd.read_csv(REDDIT_COMMENTS, encoding='unicode_escape')
        df = pd.concat([df1,df2,df3])
        df = df.sample(frac = 1)
        test_size = 0.01


    elif (website == 'redditcomments_pol_balanced'):
        path = REDDIT_COMMENTS_POL_BALANCED
        df = pd.read_csv(path)
        df = df.sample(frac = 1)
    else:
        df = pd.read_csv(website, encoding='unicode_escape')



    df['text'] = df['text'].apply(normalize)
    df = df.dropna()
    
    if separate_websites:
        df_test =  pd.concat([df[(df['website'] == 'colorlines')], df[df['website'] == 'mrc' ]]) 
        df = df[(df['website'] != 'colorlines')]
        df = df[(df['website'] != 'mrc')]

    else:
        if test_set:
            df ,df_test = train_test_split(df, random_state=1, test_size=test_size, stratify = df['class_id'])
        else:
            return  df['text'].tolist(), df['class_id'].tolist(), None, None

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


def load_data(dset_name='political_text', path = '', test_set = True):
    if dset_name == 'political_text':
        return load_pol_data()
    elif dset_name == 'disagreement':
        return load_disagreement_data()
    elif dset_name == 'ideology_fb':
        return load_ideology_data('facebook',test_set = test_set)
    elif dset_name == 'ideology_youtube':
        return load_ideology_data('youtube', test_set =test_set)
    elif dset_name == 'ideology_redditcomments':
        return load_ideology_data('redditcomments', test_set =test_set)
    elif dset_name == 'ideology_redditcomments_pol_balanced':
        return load_ideology_data('redditcomments_pol_balanced', test_set =test_set)
    elif dset_name == 'ideology_facebook_youtube':
        return load_ideology_data('youtube_facebook', test_set =test_set)
    elif dset_name == 'ideology_all':
        return load_ideology_data('all', test_set =test_set)
    elif dset_name == 'ideology_custome':
        return load_ideology_data(path, test_set)

    else:
        raise NameError(
            'Dataset not known. Available Datasets: political_text')

if __name__ == '__main__':
    print(len(load_data(dset_name = 'ideology_redditcomments_pol_balanced')[1]),len(load_data(dset_name = 'ideology_youtube')[0]),
    len(load_data(dset_name = 'ideology_youtube')[2]),len(load_data(dset_name = 'ideology_youtube')[3]))