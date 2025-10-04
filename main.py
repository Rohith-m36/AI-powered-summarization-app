import validators
import streamlit as st
import os
import subprocess
import re
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.schema import Document

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("🚨 No Groq API key found! Please set GROQ_API_KEY in your .env file.")
    st.stop()

# -----------------------------
# Streamlit Dashboard UI
# -----------------------------
st.set_page_config(page_title="Professional Summarizer Dashboard", page_icon="🦜", layout="wide")
st.title("🦜 Professional Summarizer Dashboard")
st.markdown(
    "<p style='color:#6C757D;'>Summarize YouTube videos or websites with clean, structured output and important links only.</p>",
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar Options
# -----------------------------
st.sidebar.header("⚙️ Summary Options")
summary_length = st.sidebar.slider("Summary length (approx. words)", 100, 1000, 300)
show_links = st.sidebar.checkbox("Include important links only", value=True)
summary_style = st.sidebar.selectbox("Summary style", ["Bullet Points", "Numbered List", "Paragraph"])
generic_url = st.text_input("Enter YouTube or Website URL")

# -----------------------------
# Initialize LLM
# -----------------------------
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

# -----------------------------
# Prompt Templates
# -----------------------------
map_prompt = PromptTemplate(
    template=f"""
Summarize the following content in a {summary_style.lower()}. Highlight key points, important facts, or actionable items.

Content:
{{text}}

Summary:
""",
    input_variables=["text"]
)

combine_prompt = PromptTemplate(
    template=f"""
You are an expert summarizer. Combine the following text summaries into a single, well-structured, readable summary.
- Use headings for main topics
- Format as {summary_style.lower()}
- Keep it concise and easy to read
- Make it around {summary_length} words
- After the summary, list all **important URLs/links** found (ignore social media or irrelevant links)

Summaries:
{{text}}

Final Structured Summary with Links:
""",
    input_variables=["text"]
)

# -----------------------------
# Helper function to filter important links
# -----------------------------
def filter_links(links):
    filtered = []
    for link in links:
        if any(social in link.lower() for social in ["facebook.com", "instagram.com", "twitter.com", "linkedin.com", "youtube.com"]):
            continue
        filtered.append(link.strip(".,)"))
    return filtered

# -----------------------------
# Cached Document Loader
# -----------------------------
@st.cache_data
def load_docs(url):
    docs = None
    if "youtube.com" in url or "youtu.be" in url:
        try:
            loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
            docs = loader.load()
        except Exception:
            try:
                result = subprocess.run(
                    ["yt-dlp", "--get-title", "--get-description", url],
                    capture_output=True,
                    text=True
                )
                yt_text = result.stdout.strip()
                if yt_text:
                    docs = [Document(page_content=yt_text, metadata={"source": url})]
            except Exception as yt_err:
                st.error(f"❌ Failed to extract YouTube content: {yt_err}")
    else:
        try:
            loader = UnstructuredURLLoader(
                urls=[url],
                ssl_verify=False,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) "
                                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                                  "Chrome/116.0.0.0 Safari/537.36"
                }
            )
            docs = loader.load()
        except Exception as web_err:
            st.error(f"❌ Failed to load website content: {web_err}")
    return docs

# -----------------------------
# Summarization Logic
# -----------------------------
if st.button("🦜 Generate Summary"):
    if not generic_url.strip():
        st.error("⚠️ Please enter a YouTube or Website URL")
    elif not validators.url(generic_url):
        st.error("⚠️ Invalid URL format. Please provide a valid URL.")
    else:
        try:
            with st.spinner("🔄 Loading and summarizing content..."):
                docs = load_docs(generic_url)

                if docs:
                    # Extract links
                    all_text = " ".join([doc.page_content for doc in docs])
                    links = re.findall(r'https?://\S+', all_text)
                    if show_links:
                        links = filter_links(links)

                    # Summarization chain
                    chain = load_summarize_chain(
                        llm,
                        chain_type="map_reduce",
                        map_prompt=map_prompt,
                        combine_prompt=combine_prompt
                    )

                    # ✅ Use .invoke instead of .run
                    result = chain.invoke({"input_documents": docs})
                    output_summary = result["output_text"]

                    # --------- Tabs for Dashboard ----------
                    tab1, tab2, tab3 = st.tabs(["📄 Summary", "🔗 Links", "📊 Stats"])

                    with tab1:
                        st.markdown(
                            f"<div style='background-color:#f0f4f8; padding:15px; border-radius:10px;'>"
                            f"<h3 style='color:#4B8BBE;'>Summary</h3>"
                            f"<div style='color:#333;'>{output_summary}</div></div>",
                            unsafe_allow_html=True
                        )
                        st.download_button("📥 Download Summary", output_summary, "summary.txt")

                    with tab2:
                        if show_links and links:
                            st.markdown(
                                "<div style='background-color:#e8f0fe; padding:15px; border-radius:10px;'>"
                                "<h4 style='color:#1A73E8;'>Important Links</h4>", unsafe_allow_html=True
                            )
                            for link in links:
                                st.markdown(f"- {link}")
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.info("No important links found or link display disabled.")

                    with tab3:
                        word_count = len(output_summary.split())
                        link_count = len(links) if links else 0
                        st.markdown(
                            f"<div style='background-color:#fff4e5; padding:15px; border-radius:10px;'>"
                            f"<h4 style='color:#FF8C00;'>Content Stats</h4>"
                            f"<p>Summary Word Count: <b>{word_count}</b></p>"
                            f"<p>Number of Important Links: <b>{link_count}</b></p></div>",
                            unsafe_allow_html=True
                        )
                else:
                    st.error("❌ No content found. Try another link.")

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
