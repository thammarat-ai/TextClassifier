# core packages
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import altair as alt
from datetime import datetime

# process Thai words
from pythainlp.corpus.common import thai_stopwords
from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import model_selection, preprocessing, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier

import pickle
from pathlib import Path
import streamlit_authenticator as stauth

# --- Authentication ---
names = ["Peter Parker", "Bruce Wayne", "Clark Kent", "Tony Stark"]
usernames = ["spiderman", "batman", "superman", "ironman"]


# Load hashed passwords from file
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
    "sales_dashboard", "abcdef", cookie_expiry_days=30)

name, authentication_status,usernames = authenticator.login("เข้าสู่ระบบ","main")

if authentication_status == False:
    st.error("คุณไม่ได้รับอนุญาตให้เข้าถึงหน้านี้")

if authentication_status == None:
    st.warning("กรุณาเข้าสู่ระบบ")

if authentication_status == True:

     # --- sidebar ---
    authenticator.logout("ออกจากระบบ","sidebar")
    st.sidebar.title(f"Welcome {name}")
    # Add a selectbox to the sidebar:


    # feature_extraction using CountVector
    cvec = CountVectorizer(analyzer=lambda x:x.split(' '))


    thai_stopwords = list(thai_stopwords())
    #cleansing unused
    def text_process(text, stopwords=thai_stopwords):
        if not isinstance(stopwords, list):
            raise TypeError("stopwords must be a list")
            
        text = "".join(u for u in text if u not in ("?", ".", ";", ":", "!", '"', "ๆ", "ฯ"))
        
        words = word_tokenize(text)

        words = " ".join(word for word in words)

        # remove stopwords
        processed_text = " ".join(word for word in words.split() 
                        if word not in thai_stopwords)
        return processed_text


    # load model
    with open('model_lr' , 'rb') as f:
        lr = pickle.load(f)


    def main():
        menu = ["Home", "Manage","About"]
        choice = st.sidebar.selectbox("Menu", menu)
        
        if choice == "Home":
            st.subheader("Home")
            
            with st.form(key='mlform'):
                col1, col2 = st.columns([2,1])
                with col1:
                    # message = st.text_area("บันทึกข้อความงานที่ได้ทำ")
                    message = st.text_area("บันทึกงานที่ได้รับมอบหมาย", "พานิสิตไปดูงาน", height=200)
                    submit_message = st.form_submit_button(label='ตัดคำ')
                with col2:
                    st.write("AI ช่วยวิเคราะห์งานที่ทำเป็นงานงานฝ่ายบุคลากร")
                    st.write("จะทำนายว่าเป็นงานฝ่ายบุคคลหรือ อื่นๆ")
                
                
            if submit_message:
                my_text = message
                my_tokens = text_process(my_text)
                # st.write(my_tokens)
                # my_tokens = [my_tokens]
                my_bow = cvec.transform(pd.Series([my_tokens]))
                my_predictions = lr.predict(my_bow)


                # prediction = model.predict_one(message)
                # prediction_proba = model.predict_proba_one(message)
                # probability = max(prediction_proba.values())
                # postdate = datetime.now()
                #add data to db
                # add_data(message, prediction, probability, prediction_proba['software'], prediction_proba['hardware'], postdate)
                st.success("Data Submitted")

                st.write("Prediction: ", my_predictions)

                # res_col1, res_col2 = st.columns(2)
                # with res_col1:
                #     st.info("Original Text")
                #     st.write(message)

                #     st.success("Prediction")
                #     st.write(prediction)

                # with res_col2:
                #     st.info("Probability")
                #     st.write(prediction_proba)

                #     #Plot of Probability
                #     df_proba = pd.DataFrame({'label':prediction_proba.keys(), 'probability':prediction_proba.values()})
                #     st.dataframe(df_proba)
                #     # visualize
                #     fig = alt.Chart(df_proba).mark_bar().encode(x='label', y='probability')
                #     st.altair_chart(fig, use_container_width=True)



        elif choice == "Manage":
            st.subheader("Manage")
            # stored_data = view_all_data()
            # new_df = pd.DataFrame(stored_data, columns=['message', 'prediction', 'probability', 'software_proba', 'hardware_proba', 'postdate'])
            # st.dataframe(new_df)
            # new_df['postdate'] = pd.to_datetime(new_df['postdate'])
            # # c = alt.Chart(new_df).mark_line().encode(x='minutes(postdate)', y='probability')
            # c = alt.Chart(new_df).mark_line().encode(x='postdate', y='probability')
            # st.altair_chart(c, use_container_width=True)
            
            # c_software_proba = alt.Chart(new_df['software_proba'].reset_index()).mark_line().encode(x='software_proba', y='index')
            # c_hardware_proba = alt.Chart(new_df['hardware_proba'].reset_index()).mark_line().encode(x='hardware_proba', y='index')
            
            # # st.altair_chart(c_software_proba)
            # c1, c2 = st.columns(2)
            # with c1:
            #     with st.expander("Software Probability"):
            #         st.altair_chart(c_software_proba,use_container_width=True)
            # with c2:    
            #     with st.expander("Hardware Probability"):
            #         st.altair_chart(c_hardware_proba,use_container_width=True)
            # with st.expander("Prediction Distribution"):
            #     fig2 = plt.figure()
            #     sns.countplot(x='prediction', data=new_df)
            #     st.pyplot(fig2)


        else:
            st.subheader("About")
            st.write('This app is built by gig')

    if __name__ == '__main__':
        main()

        
