import streamlit as st
from sentence_transformers import SentenceTransformer
import requests
import pandas as pd

st.set_page_config(layout='wide')

@st.cache(allow_output_mutation=True)
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

if __name__ == '__main__':
    encoder = load_model()

    k = st.number_input('Number of results to return', value=30, max_value=200)

    st.caption(
        'Enter your query in the textbox below. Try geo-scoping your results by including location '
        'names in your query e.g. "Texas new street light replacements".')
    
    query = st.text_input('Search Query')

    if query != "":
        q = encoder.encode(query, normalize_embeddings=True)
        res = requests.post(st.secrets['URL'], json={'q':q.tolist(), 'k':k})

        for r in res.json():
            st.markdown('#### '+ r['title'])
            st.caption('media_item_id: {}; cosine similarity: {:0.04f}'.format(r['media_item_id'], r['cos_sim']))
            with st.expander('content'):
                st.markdown(r['content'])
            st.text(' \n \n')


