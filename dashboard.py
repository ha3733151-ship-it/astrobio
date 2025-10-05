import pandas as pd
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="NASA Bioscience Dashboard", layout="wide", page_icon="🧬")

# ========= Load Data =========
file_path = r"D:\NASA\NASA_summarized.xlsx"
try:
    df = pd.read_excel(file_path)
except Exception:
    df = pd.read_csv(file_path, encoding="utf-8", errors="ignore")

for col in ['Title', 'Authors', 'Journal', 'Year', 'Abstract', 'Summary', 'Link']:
    if col not in df.columns:
        df[col] = None

df['Year'] = df['Year'].astype(str).str.strip()

# ========= CSS Styling =========
st.markdown("""
<style>
body {
    background-color: #0b1120;
    font-family: 'Segoe UI', sans-serif;
    color: white;
}
.sidebar-filter {
    background: linear-gradient(180deg, #0b1120, #1e1b4b);
    border-radius: 25px;
    padding: 25px 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    color: white;
    text-align: center;
}
.sidebar-filter h3 {
    color: #c084fc;
    font-size: 30px;
    font-weight: bold;
    margin-bottom: 20px;
    letter-spacing: 1px;
}
.sidebar-filter label {
    color: #000000;
    font-size: 25px;
    font-weight: 600;
}
div[data-baseweb="select"], .stTextInput, .stNumberInput, .stDateInput input {
    background-color: #a855f7 !important;
    border-radius: 12px !important;
    color: black !important;
    border: 5px solid #c084fc !important;
}
.stSelectbox [data-baseweb="select"] div {
    color: black !important;
}
.stSelectbox label, .stTextInput label {
    color: #000000 !important;
    font-size: 25px
}
.chatbot-section {
    background: linear-gradient(180deg, #0b1120, #1e1b4b);
    border-radius: 25px;
    padding: 25px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    color: white;
    text-align: center;
}
.chatbot-section h3 {
    color: #c084fc;
    font-size: 22px;
    font-weight: bold;
    margin-bottom: 15px;
}
.chatbot-section textarea {
    background-color: #a855f7 !important;
    color: white !important;
    border-radius: 12px !important;
    border: 2px solid #c084fc !important;
    padding: 10px !important;
    font-size: 14px !important;
}
.stButton>button {
    background-color: #a855f7;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 8px 20px;
    font-size: 15px;
    font-weight: bold;
    box-shadow: 0 3px 8px rgba(0,0,0,0.3);
    transition: 0.2s;
}
.stButton>button:hover {
    background-color: #c084fc;
    transform: scale(1.05);
}
.data-card {
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(200, 132, 252, 0.3);
    border-radius: 20px;
    padding: 20px;
    margin-bottom: 20px;
    color: white;
    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
}
.data-card h4, .data-card h3 {
    color: #c084fc;
}
h1, h2, h3, h4, h5 {
    color: #c084fc !important;
}
hr, .stMarkdown > div > hr {
    border: 1px solid rgba(255,255,255,0.1);
}
</style>
""", unsafe_allow_html=True)

# ========= Layout =========
col1, col2, col3 = st.columns([1.2, 3.5, 1.2])

# ========= Sidebar Filters =========
with col1:
    st.markdown('<div class="sidebar-filter">', unsafe_allow_html=True)
    st.markdown("### 🔍 Filters")

    all_authors = df['Authors'].dropna().astype(str).str.split(',').explode().str.strip().unique()
    selected_author = st.selectbox("👤 Author", ["All"] + sorted(all_authors))

    all_years = sorted(df['Year'].dropna().unique())
    selected_year = st.selectbox("📅 Year", ["All"] + list(all_years))

    filtered_titles_df = df.copy()
    if selected_author != "All":
        filtered_titles_df = filtered_titles_df[filtered_titles_df['Authors'].astype(str).str.contains(selected_author, na=False)]
    if selected_year != "All":
        filtered_titles_df = filtered_titles_df[filtered_titles_df['Year'] == selected_year]

    all_titles = filtered_titles_df['Title'].dropna().unique()
    selected_title = st.selectbox("📄 Title", ["All"] + list(all_titles))

    st.markdown('</div>', unsafe_allow_html=True)

# ========= Apply Filters =========
filtered_df = df.copy()
if selected_author != "All":
    filtered_df = filtered_df[filtered_df['Authors'].astype(str).str.contains(selected_author, na=False)]
if selected_year != "All":
    filtered_df = filtered_df[filtered_df['Year'] == selected_year]
if selected_title != "All":
    filtered_df = filtered_df[filtered_df['Title'] == selected_title]

# ========= Main Content =========
with col2:
    st.markdown("<h1 style='text-align: center;'>🧬 NASA Bioscience Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Explore 600+ NASA bioscience publications by author, year, and title.</p>", unsafe_allow_html=True)
    st.write("---")

    if selected_author != "All" or selected_year != "All" or selected_title != "All":

        # ===== Knowledge Graph =====
        st.markdown("### 🧠 Knowledge Graph: Authors & Publications")
        if not filtered_df.empty:
            G = nx.Graph()
            for idx, row in filtered_df.iterrows():
                title = row['Title']
                authors = row['Authors']
                if pd.isna(title) or pd.isna(authors):
                    continue
                author_list = [a.strip() for a in authors.split(',')]
                G.add_node(title, type='publication')
                for author in author_list:
                    G.add_node(author, type='author')
                    G.add_edge(author, title)

            plt.figure(figsize=(10, 6))
            pos = nx.spring_layout(G, k=0.5)
            color_map = ['#60a5fa' if G.nodes[n]['type']=='author' else '#86efac' for n in G]
            nx.draw(G, pos, with_labels=True, node_color=color_map, node_size=500, font_size=8)
            st.pyplot(plt)
            plt.clf()
        else:
            st.info("No data found for graph generation.")

        # ===== Publications Section =====
        st.markdown("### 📄 Publications")
        for _, row in filtered_df.iterrows():
            st.markdown('<div class="data-card">', unsafe_allow_html=True)
            st.markdown(f"#### {row['Title']}")
            st.markdown(f"*Authors:* {row['Authors']}")
            st.markdown(f"*Journal:* {row['Journal']}")
            st.markdown(f"*Year:* {row['Year']}")
            st.markdown(f"*Abstract:* {row['Abstract']}")
            st.markdown(f"*Summary:* {row['Summary']}")
            if pd.notna(row['Link']):
                st.markdown(f"[🔗 Read Full Paper]({row['Link']})")
            st.markdown('</div>', unsafe_allow_html=True)

        # ===== Related Publications Section =====
        if not filtered_df.empty:
            st.markdown("### 🔗 Related Publications")
            try:
                corpus = df['Abstract'].fillna("").tolist()
                vectorizer = TfidfVectorizer(stop_words="english")
                X = vectorizer.fit_transform(corpus)

                if selected_title != "All":
                    base_indices = [df.index[df['Title'] == selected_title][0]]
                else:
                    base_indices = filtered_df.index.tolist()

                related_idx = set()
                for idx in base_indices:
                    cosine_sim = cosine_similarity(X[idx], X).flatten()
                    top_related = cosine_sim.argsort()[-6:][::-1]
                    top_related = [i for i in top_related if i != idx]
                    related_idx.update(top_related)

                for i in list(related_idx)[:10]:
                    st.markdown('<div class="data-card">', unsafe_allow_html=True)
                    st.markdown(f"**{df.iloc[i]['Title']}**")
                    st.caption(f"Authors: {df.iloc[i]['Authors']}")
                    st.caption(f"Year: {df.iloc[i]['Year']}")
                    st.write(df.iloc[i]['Abstract'][:400] + "...")
                    if pd.notna(df.iloc[i]['Link']):
                        st.markdown(f"[🔗 Read Full Paper]({df.iloc[i]['Link']})")
                    st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.warning(f"Couldn't fetch related publications: {e}")
    else:
        st.info("Please select filters to display publications and knowledge graph.")

# ========= Chatbot Section =========
with col3:
    st.markdown('<div class="chatbot-section">', unsafe_allow_html=True)
    st.markdown("### 🤖 Astro Chat Assistant")
    st.write("Ask anything about NASA bioscience data:")

    user_input = st.text_area("Type your question here:", placeholder="e.g., show me papers about plants")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if st.button("Ask"):
        query = user_input.strip()
        if query == "":
            st.warning("Please enter a question.")
        else:
            df['combined_text'] = df[['Title', 'Abstract', 'Summary']].fillna('').agg(' '.join, axis=1)
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
            query_vec = vectorizer.transform([query])
            similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
            top_indices = similarity_scores.argsort()[::-1][:5]
            top_results = df.iloc[top_indices][similarity_scores[top_indices] > 0]

            if not top_results.empty:
                st.session_state.chat_history.append(("🧑 You", query))
                for idx, row in top_results.iterrows():
                    answer = f"{row['Title']}\n\n{row['Summary']}\n\n[Read More]({row['Link']})" if pd.notna(row['Link']) else row['Summary']
                    st.session_state.chat_history.append(("🤖 AstroBot", answer))
            else:
                st.session_state.chat_history.append(("🧑 You", query))
                st.session_state.chat_history.append(("🤖 AstroBot", "No relevant publications found."))

    for sender, msg in reversed(st.session_state.chat_history[-10:]):
        st.markdown(f"{sender}:** {msg}")
        st.write("---")

    st.markdown('</div>', unsafe_allow_html=True)
