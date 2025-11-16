import streamlit as st
import pywhatkit as pwk
import datetime

st.title("💬 WhatsApp Message Scheduler")

# Input fields
phone_number = st.text_input("Enter phone number (with country code)", "+91")
message = st.text_area("Enter your message")
send_hour = st.number_input("Hour (24-hour format)", min_value=0, max_value=23, step=1)
send_minute = st.number_input("Minute", min_value=0, max_value=59, step=1)

if st.button("Schedule Message"):
    if phone_number and message:
        try:
            # Schedule the message
            st.success(f"Message scheduled for {send_hour}:{send_minute:02d}")
            pwk.sendwhatmsg(phone_number, message, send_hour, send_minute)
            st.success("Message sent successfully!")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please fill all fields.")
