import pathlib 
 
files = ['streamlit_app/pages/1_predict.py','streamlit_app/pages/2_analytics.py','streamlit_app/pages/3_history.py','streamlit_app/pages/5_ai_assistant.py'] 
old = 'st.session_state.get("api_key", "dev-secret-key")' 
new = 'st.session_state.get("api_key", os.getenv("API_KEY", "dev-secret-key"))' 
for f in files: 
    p = pathlib.Path(f) 
    content = p.read_text(encoding='utf-8') 
    p.write_text(content.replace(old, new), encoding='utf-8') 
    print('Fixed:', f) 
