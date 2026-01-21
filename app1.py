# import required libraries
import streamlit as st
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_google_genai import ChatGoogleGenerativeAI

## Page setup
st.set_page_config(
    page_title="Diabetes SQL Chatbot",
    layout="centered"
)

st.title("🩺 Diabetes Patient SQL Chatbot")
st.write("Ask questions about patient sugar levels")

# Database
engine = create_engine("sqlite:///diabetes.db")
db = SQLDatabase(engine)

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0,
    google_api_key=st.secrets["GOOGLE_API_KEY"]
)

# create sql agent that generate query and execute them on database
agent = create_sql_agent(
    llm=llm,
    db=db,
    verbose=True,
    agent_executor_kwargs={
        "handle_parsing_errors": True, # prevent crashes
        "return_intermediate_steps": True   
    }
)


# User input
question = st.text_input(
    "Ask your question:",
    placeholder="Who has the highest sugar level?"
)

# Button action
if st.button("Get Answer") and question:
    try:
        # invoke the agent with natural language question
        response = agent.invoke({"input": question})

        st.subheader("Answer")
        st.success(response["output"])

        
        if "intermediate_steps" in response:
            st.subheader("SQL Query and Database Output")
            st.subheader("SQL Query & Database Output")

            for i, step in enumerate(response["intermediate_steps"], start=1):
                action, observation = step

                st.markdown(f"**Step {i}: Tool Used → `{action.tool}`**")

                
                if isinstance(action.tool_input, str) and "SELECT" in action.tool_input.upper():
                    st.markdown("**Generated SQL Query:**")
                    st.code(action.tool_input, language="sql")

                # display the database output from the executed query
                if observation:
                    st.markdown("**Database Output:**")
                    st.write(observation)

    except Exception as e:
        st.error(f"Error: {e}")


