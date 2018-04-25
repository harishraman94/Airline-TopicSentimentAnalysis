
import cleaning
import codecs
import pandas as pd
import subprocess
def main():
    doc_clean,document_tweets=cleaning.get_clean_docs(0)
    file = codecs.open('data.txt', 'w', 'utf-8')
    doc_list=[]
    print (len(doc_clean))
    print(len(document_tweets))
    for doc in doc_clean:
        document_text=" ".join(doc)
        file.write(document_text+'\n')

    p = subprocess.Popen(["sh", "runExample.sh"], shell=True)

    print(p.communicate())
    result_doc=open("C:/Users/brohi/OneDrive/Desktop/BTM-master/output/model/k5.pz_d",'r')
    topic=[]
    for doc in result_doc.readlines():
        doc_dist=doc.split()
        topic.append(doc_dist.index(max(doc_dist)))

    df = pd.DataFrame({'text':document_tweets[:len(topic )]})

    se = pd.Series(topic)
    df['label'] = se
    df.to_csv('tested_btm.csv')
    df_temp =df[df['label']==0]
    df_temp.to_csv('tested_btm1.csv')
    df_temp=df[df['label']==1]
    df_temp.to_csv('tested_btm2.csv')
    df_temp=df[df['label']==2]
    df_temp.to_csv('tested_btm3.csv')
    df_temp=df[df['label']==3]
    df_temp.to_csv('tested_btm4.csv')
    df_temp=df[df['label']==4]
    df_temp.to_csv('tested_btm5.csv')


def evaluateFordict(eval_temp,df):
    cout=0
    for index,col in df.iterrows():
            cout += 1
    print ('acc :',(cout/df['negativereason'].value_counts(dropna=True).sum()))
    return cout/df.shape[0]

def eval(df):
    eval_temp={
    '0'  :['Cancelled Flight'],
    '1' : ['Late Flight'],
    '2' : ['Lost Luggage','Damaged Luggage','Flight Attendant Complaints','Flight Booking Problems'],
    '3' : ['Customer Service Issue'],
    '4' :['Can\'t Tell','longlines']
    }
    evaluateFordict(eval_temp,df)

main()
