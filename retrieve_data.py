import re
import pandas as pd


def clean_text(txt):
    txt = txt.lower()
    txt = re.sub(r"i'm", "i am", txt)
    txt = re.sub(r"he's", "he is", txt)
    txt = re.sub(r"she's", "she is", txt)
    txt = re.sub(r"that's", "that is", txt)
    txt = re.sub(r"what's", "what is", txt)
    txt = re.sub(r"where's", "where is", txt)
    txt = re.sub(r"\'ll", " will", txt)
    txt = re.sub(r"\'ve", " have", txt)
    txt = re.sub(r"\'re", " are", txt)
    txt = re.sub(r"\'d", " would", txt)
    txt = re.sub(r"won't", "will not", txt)
    txt = re.sub(r"can't", "can not", txt)
    txt = re.sub(r"[^\w\s]", "", txt)
    return txt


def cornell(path='data/cornell_dialogue_datasets'):
    lines = open(path + '/movie_lines.txt', encoding='utf-8',
                 errors='ignore').read().split('\n')

    convers = open(path + '/movie_conversations.txt',
                   encoding='utf-8',
                   errors='ignore').read().split('\n')

    exchn = []

    for conver in convers:
        exchn.append(conver.split(' +++$+++ ')[-1][1:-1].replace("'", " ").replace(",", "").split())

    diag = {}

    for line in lines:
        diag[line.split(' +++$+++ ')[0]] = line.split(' +++$+++ ')[-1]

    questions = []
    answers = []

    for conver in exchn:
        for i in range(len(conver) - 1):
            questions.append(diag[conver[i]])
            answers.append(diag[conver[i + 1]])

    ###############################
    #        max_len = 13         #
    ###############################

    sorted_ques = []
    sorted_ans = []

    for i in range(len(questions)):
        if len(questions[i]) < 13:
            sorted_ques.append(questions[i])
            sorted_ans.append(answers[i])

    clean_ques = []
    clean_ans = []

    for line in sorted_ques:
        clean_ques.append(clean_text(line))

    for line in sorted_ans:
        clean_ans.append(clean_text(line))

    for i in range(len(clean_ans)):
        clean_ans[i] = ' '.join(clean_ans[i].split()[:11])

    ## trimming
    clean_ans = clean_ans[:30000]
    clean_ques = clean_ques[:30000]

    return (clean_ans, clean_ques)



def himym(path='data/HIMYM'):
    df = pd.read_csv(path + '/HIMYM.txt', delimiter='+')
    df['Line'] = df['Line'].apply(clean_text)

    barney_lines = list(df[df.Character.isin(['Barney', 'BARNEY'])].index)
    '''
    ted_lines = list(df[df.Character.isin(['Ted', 'TED'])].index)
    marshall_lines = list(df[df.Character.isin(['Marshall', 'MARSHALL'])].index)
    lily_lines = list(df[df.Character.isin(['Lily', 'LILY'])].index)
    robin_lines = list(df[df.Character.isin(['Robin', 'ROBIN'])].index)
    '''
    utterances = []
    responses = []

    for _, Scene_grp in df.groupby('Scene'):
        utterance = ''
        for row in Scene_grp.values:
            if (row[0] + 1) in barney_lines:
                utterances.append(utterance.split('+')[-1])
                utterance = ''
                responses.append(row[-2])
            else:
                utterance = utterance + '+' + row[-2]

    return (utterances, responses)
