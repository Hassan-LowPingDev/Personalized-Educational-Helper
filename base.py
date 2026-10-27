
import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

load_dotenv()
st.set_page_config(page_title="Support AI", layout="wide")

class TicketAnalysis(BaseModel):
    category: str = Field(description="Billing, Delivery, Technical, or Other")
    sentiment: str = Field(description="Positive, Neutral, or Negative")
    priority: str = Field(description="Low, Medium, or High")
    root_cause: str
    recommended_action: str

st.title("🎧 AI Customer Support Investigator")
st.caption("Turn customer complaints into actionable business insights.")

uploaded = st.file_uploader("Upload a CSV of customer tickets", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = pd.DataFrame({
        "ticket_id": [1, 2, 3, 4],
        "customer_message": [
            "I was charged twice for my order!",
            "My package is five days late.",
            "The app crashes whenever I log in.",
            "Thank you, the replacement arrived."
        ]
    })

st.subheader("Customer tickets")
st.dataframe(df, use_container_width=True)

if st.button("Analyze tickets", type="primary"):
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Add your GOOGLE_API_KEY to the .env file.")
        st.stop()

    llm = init_chat_model(
    model="google_genai:gemini-3.8-flash",
    temperature=0
    )
    analyzer = llm.with_structured_output(TicketAnalysis)

    results = []
    progress = st.progress(0)
    status = st.empty()

    try:
        for i, row in df.iterrows():
            status.write(f"Analyzing ticket {i + 1} of {len(df)}...")

            result = analyzer.invoke([
                ("system", """
                Classify the customer ticket.
                Be cautious when assigning high priority.
                Only use the supplied message as evidence.
                """),
                ("human", str(row["customer_message"]))
            ])

            item = row.to_dict()
            item.update(result.model_dump())
            results.append(item)
            progress.progress((i + 1) / len(df))

        st.session_state["results"] = pd.DataFrame(results)
        status.success("Analysis complete!")

    except Exception as e:
        st.error(f"Analysis failed: {e}")

if "results" in st.session_state:
    result_df = st.session_state["results"]

    st.subheader("Business insights")
    a, b, c = st.columns(3)
    a.metric("Total tickets", len(result_df))
    b.metric(
        "Negative sentiment",
        int((result_df["sentiment"] == "Negative").sum())
    )
    c.metric(
        "High priority",
        int((result_df["priority"] == "High").sum())
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.pie(result_df, names="category",
                   title="Complaints by category"),
            use_container_width=True
        )
    with right:
        st.plotly_chart(
            px.bar(result_df, x="priority",
                   title="Tickets by priority"),
            use_container_width=True
        )

    st.subheader("Investigated tickets")
    st.dataframe(result_df, use_container_width=True)

    selected = st.selectbox(
        "Choose a ticket to draft a response",
        result_df.index,
        format_func=lambda i: str(
            result_df.iloc[i]["customer_message"]
        )[:80]
    )

    if st.button("Generate customer reply"):
        llm = init_chat_model(
        model="google_genai:gemini-3.8-flash",
        temperature=0
        )
        ticket = result_df.iloc[selected]

        reply = llm.invoke([
            ("system", """
            Write a concise, empathetic customer support reply.
            Never claim a refund, replacement, or other action
            has already happened unless the supplied facts confirm it.
            """),
            ("human", str(ticket["customer_message"]))
        ])
        st.write(reply.content)

    st.download_button(
        "Download analyzed tickets",
        result_df.to_csv(index=False),
        "analyzed_tickets.csv",
        "text/csv"
    )
