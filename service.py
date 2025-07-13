import streamlit as st 

pages = {
    "Information":[
        st.Page("about_page.py", title = "About")
    ],
    "Projects": [
        st.Page("SRGAN/srgan_page.py", title="SRGAN", icon="🖼️"),
        st.Page("SAE/sae_page.py", title="SAE", icon="🧠"),
    ]
    
}

pg = st.navigation(pages)
pg.run()