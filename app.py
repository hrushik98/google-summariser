
import openai
import streamlit as st
from googlesearch import search
import requests
session = requests.Session()
session.headers['User-Agent']
my_headers = {"User-Agent": "Mozilla/5.0 (X11; CrOS x86_64 14685.0.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.4992.0 Safari/537.36",
              "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9"}
from bs4 import BeautifulSoup as bs
question = st.text_input("Enter your query")
st.header("Google summarizer")
st.write("Get the summary from top 10 links of google")
if st.button("Go"):
    openai.api_key = st.secrets['API_KEY']
    con = ""
    urls = []
    passages = []
    query = question
    for j in search(query, num=10, stop=10, pause=2):
        urls.append(j)
    
    file_name = "text-data"
    for url in urls:
        result = session.get(url, headers = my_headers)
    
    doc = bs(result.content, "html.parser")
    contents = doc.find_all("p")
    for content in contents:
        passages.append(content.text)
        con = con + content.text + "\n"
    with open('./{}.txt'.format(file_name), mode='wt', encoding='utf-8') as file:
        file.write(con)
    # #the edited code
    # passages2 = []
    # i = 0
    # for x in range(1,10):
    #     i = i
    #     Z = ""
    #     P = ""
    # while len(Z) <=60:
    #     P += (passages[i])
    #     Z = P.split()
    #     i+=1
    # passages.append(P)
    from rank_bm25 import BM25Okapi
    from sklearn.feature_extraction import _stop_words
    import string
    from tqdm.autonotebook import tqdm
    import numpy as np

    def bm25_tokenizer(text):
        tokenized_doc = []
        for token in text.lower().split():
            token = token.strip(string.punctuation)

            if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
                tokenized_doc.append(token)
        return tokenized_doc

    tokenized_corpus = []
    for passage in tqdm(passages):
        tokenized_corpus.append(bm25_tokenizer(passage))

    bm25 = BM25Okapi(tokenized_corpus)

    bm25_scores = bm25.get_scores(bm25_tokenizer(query))
    top_n = np.argpartition(bm25_scores, -10)[-10:]
    bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
    bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)

    bm25_passages = []
    for hit in bm25_hits:
        bm25_passages.append(passages[hit["corpus_id"]])
    

    response = openai.Completion.create(
        
        model="text-davinci-003",
        prompt = "Imagine an AI agent that can generate human-like text based on the Customer query on the Dialogue History and Supporting Texts. The information on the Supporting Texts can be used to reinforce the AI's response. It can provide information on a wide range of topics, answer questions, and engage in conversation on a variety of subjects including history, science, literature, art, and current events. It can provide information on basic facts to more complex topics. If one has a question about a specific topic, It'll do it's best to provide a relevant and accurate response. It can also generate text in a variety of styles and formats, depending on the task at hand. It can write narratives, descriptions, articles, reports, letters, emails, poems, stories and many other types of texts. \n\n###\n\nDialogue History:\nCustomer: "+str(query)+"\n\nSupporting Texts:\nSupporting Text 1: "+str(bm25_passages[0])+"\nSupporting Text 2: "+str(bm25_passages[1])+"\nSupporting Text 3: "+str(bm25_passages[2])+"\nSupporting Text 4: "+str(bm25_passages[3])+"\nSupporting Text 5: "+str(bm25_passages[4])+"\nSupporting Text 6: "+str(bm25_passages[5])+"\nSupporting Text 7: "+str(bm25_passages[6])+"\n\nAgent Response:",
        temperature=0.7,
        max_tokens=500,
        top_p=0.3,
        frequency_penalty=2,
        presence_penalty=0,
        stop = ['Supporting Text','Agent']
    )
    ans2 = response['choices'][0]['text']
    st.header(ans2)
