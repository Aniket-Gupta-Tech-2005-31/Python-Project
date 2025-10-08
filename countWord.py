import re
import streamlit as st

st.title("Welcome to word count streamlit app ")
st.subheader("This app counts the total number of words in a given sentence.")
text=st.text_input("Enter the sentence in the input box below")
st.write("The total number of words in the sentence is :- ",len(re.findall(r'\b\w+\b',text)))
st.write("The total number of cherecter in the sentence is :- ",len(text))

                                                                           
