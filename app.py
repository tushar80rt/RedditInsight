import streamlit as st
import pandas as pd
from agents import fetch_posts, generate_report, create_post, generate_comment_from_best, ask_helper
from dotenv import load_dotenv
import os
import traceback


# ---------------- Load environment variables ---------------- #
dotenv_path = os.path.join(os.path.dirname(__file__), "api.env")
if not os.path.exists(dotenv_path):
    st.error(f".env file not found at {dotenv_path}")
load_dotenv(dotenv_path)

# ---------------- Streamlit Page Config ---------------- #
st.set_page_config(
    page_title="https://FactCheck-agent.com",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= UI Styling =================
st.markdown("""
<style>
.stApp { background-color: #0E1117; color: #FAFAFA; }
.main-header { font-size: 2.8rem; color: #00D4B1; text-align: center; font-weight: 800; margin-bottom: 0.2rem; letter-spacing: -0.5px; }
.sub-header { text-align: center; color: #888; margin-bottom: 2.5rem; font-size: 1.1rem; }
.stButton>button { width: 100%; background-color: #555555; color: #FAFAFA; border: none; padding: 0.8rem 1rem; border-radius: 8px; font-weight: 600; font-size: 1rem; margin-top: 1rem; }
.stButton>button:hover { background-color: #777777; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
.stTextInput>div>div>input { background-color: #262730; color: #FAFAFA; border: 1px solid #393946; border-radius: 8px; padding: 0.8rem; }
.stTextInput>div>div>input:focus { border-color: #00D4B1; box-shadow: 0 0 0 2px rgba(0, 212, 177, 0.2); }
.css-1d391kg, .css-1d391kg>div { background-color: #0E1117 !important; border-right: 1px solid #262730; }
.css-1d391kg h1,h2,h3,h4,h5,h6,p,label { color: #FAFAFA !important; }
.stProgress > div > div > div > div { background-color: #00D4B1; }
.streamlit-expanderHeader { background-color: #262730; color: #FAFAFA; border-radius: 8px; font-weight: 600; }
.streamlit-expanderContent { background-color: #1A1D25; border-radius: 0 0 8px 8px; }
.card { background-color: #262730; padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem; border-left: 4px solid #00D4B1; }
.stRadio > div { background-color: #262730; padding: 1rem; border-radius: 8px; }
label { font-weight: 600 !important; margin-bottom: 0.5rem; display: block; color: #CCC !important; }
.main-title { font-size: 40px; font-weight: bold; display: flex; align-items: center; }
.main-title img { height: 50px; margin-left: 20px; vertical-align: middle; }
.subtitle { font-size: 20px; color: #AAAAAA; }
</style>
<div class="header">
  <div class="main-title">
    Reddit Insight Agents
    <img src="https://registry.npmmirror.com/@lobehub/icons-static-png/latest/files/dark/crewai-brand-color.png" alt="Camel-AI Logo">
    <span style="margin-left:40px;">&</span>
    <img src="https://redditinc.com/hubfs/Reddit%20Inc/Brand/Reddit_Lockup_Logo.svg" alt="LangGraph Logo">
  </div>
  <div class="subtitle"> Analyzing Reddit Posts for Accuracy </div>
  <br>
</div>
""", unsafe_allow_html=True)

# ---------------- Sidebar ---------------- #
col1, col2 = st.columns(2)

with col1:
    subreddits_input = st.text_input(
        label="🌐Subreddits",
        placeholder="e.g. news, science, india",
        help="Enter subreddit names without the 'r/' prefix."
    )

with col2:
    keywords_input = st.text_input(
        label="📝Keywords",
        placeholder="e.g. AI, stock market, elections",
        help="Keywords to search for in posts and comments."
    )

with st.sidebar:
    st.image("./assets/openai-text.png", width=200)
    Openai_key = st.text_input("OpenAI API key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    reddit_key = st.text_input("Reddit_client_secret", value=os.getenv("REDDIT_CLIENT_SECRET", ""), type="password")

    if st.button("💾 Save Keys"):
        st.session_state["OPENAI_API_KEY"] = Openai_key
        st.session_state["REDDIT_CLIENT_SECRET"] = reddit_key
    if Openai_key or reddit_key:
        st.success("Keys saved for this session")

    post_limit = st.number_input(
        "Posts per subreddit",
        min_value=1, max_value=20, value=5, step=1,
        help="Number of posts to fetch from each subreddit"
    )

    comment_limit = st.number_input(
        "Comments per post",
        min_value=1, max_value=10, value=3, step=1,
        help="Number of top comments to analyze per post"
    )

    st.markdown("---")
    st.subheader("✍️ Custom Post to Reddit")

    custom_subreddit = st.text_input("Subreddit for posting")
    custom_title = st.text_input("Post Title")
    custom_body = st.text_area("Post Body")

    # Flair selection (optional, fetch from Reddit if needed)
    custom_flair = st.text_input("Post Flair (if required by subreddit)", help="Some subreddits require flair. Enter flair text here.")

    post_btn = st.button("📤 Post to Reddit")

    if post_btn:
        if not custom_subreddit or not custom_title or not custom_body:
            st.error("Please fill all fields to post.")
        else:
            try:
                submission = create_post(custom_subreddit, custom_title, custom_body, flair_text=custom_flair)
                if submission:
                    st.success(f"Post created successfully! [View Post](https://reddit.com{submission.permalink})")
                else:
                    st.error("Failed to create post. Check credentials and subreddit rules.")
            except Exception as e:
                st.error(f"Error: {e}")


with st.sidebar:
    st.markdown("---")
    st.subheader("💡 Generate Comment")
    st.write("Generate a new comment based on the most liked comment from fetched posts.")

    comment_post_index = st.number_input(
        "Select Post Index",
        min_value=1, max_value=20, value=1, step=1,
        help="Choose the index of the post from which you want to generate a new comment."
    )
    gen_comment_btn = st.button("Generate Comment")

    if gen_comment_btn:
        try:
            posts_data_for_comment = fetch_posts(
                [s.strip() for s in subreddits_input.split(",") if s.strip()],
                [k.strip() for k in keywords_input.split(",") if k.strip()],
                post_limit,
                comment_limit
            )
            if not posts_data_for_comment or comment_post_index > len(posts_data_for_comment):
                st.error("Invalid post index or no posts fetched!")
            else:
                selected_post = posts_data_for_comment[comment_post_index-1]
                top_comment = generate_comment_from_best(selected_post.get("Comments", []))
                if top_comment:
                    st.success("Generated Comment:")
                    st.text_area("Your Generated Comment", value=top_comment, height=150)
                else:
                    st.info("No comments available to generate a new comment.")
        except Exception as e:
            st.error(f"Error generating comment: {e}")


    st.markdown("---")
    st.subheader("About")
    st.markdown("""
    This app helps you analyze Reddit posts and comments to identify misinformation,
    assess sentiment, and generate insightful comments.

    Key features include:
    - Automatic fetching and analysis of posts from your chosen subreddits.
    - Fact-checking and sentiment analysis for comments to identify misleading content.
    - Generation of new comments inspired by top-liked comments.
    - Ability to automatically post your generated comments or custom posts to Reddit.

    It leverages AI agents for content summarization, fact-checking, and comment generation
    to provide accurate and meaningful insights, while allowing interactive engagement with Reddit content.
    """)


# ---------------- Analysis Button ---------------- #
analyze_btn = st.button("Start Analysis", type="primary")

if analyze_btn:
    subreddits = [s.strip() for s in subreddits_input.split(",") if s.strip()]
    keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]

    if not subreddits:
        st.error("Please enter at least one valid subreddit!")
        st.stop()

    if not keywords:
        st.error("Please enter at least one valid keyword!")
        st.stop()

    progress_bar = st.progress(0)
    status_text = st.empty()

    with st.spinner("Connecting to Reddit API..."):
        posts_data = []
        try:
            status_text.text("Fetching posts from Reddit...")
            posts_data = fetch_posts(subreddits, keywords, post_limit, comment_limit)
            progress_bar.progress(50)
        except Exception as e:
            st.error("Error fetching posts! Check terminal for details.")
            st.error(f"Error: {str(e)}")
            print("Exception in fetch_posts():", e)
            traceback.print_exc()
            st.stop()

    if not posts_data:
        st.warning("No posts fetched. Check your API credentials or search parameters.")
        st.stop()

    progress_bar.progress(75)
    status_text.text("Analyzing content...")

    st.success(f"Fetched {len(posts_data)} posts successfully")

    for i, post in enumerate(posts_data[:3]):
        try:
            # Card layout
            st.markdown('<div class="card">', unsafe_allow_html=True)
            cols = st.columns([1, 3])


            with cols[0]:
                if post.get("Post Thumbnail"):
                    st.image(post["Post Thumbnail"], use_container_width=True)
                else:
                    st.image("https://www.redditinc.com/assets/images/site/reddit-logo.png", use_container_width=True)

            # Title + Content on right
            with cols[1]:
                st.markdown(f"### 🔗 [{post.get('Post Title', 'No Title')}]({post.get('Post Link')})")
                st.caption(f"r/{post.get('Subreddit', 'N/A')} • 👍 {post.get('Post Upvotes', 0)} upvotes")

                # Post body preview
                post_body = post.get("Post Body", "")
                if post_body:
                    preview_text = post_body[:250] + "..." if len(post_body) > 250 else post_body
                    st.write(preview_text)

                    with st.expander("Read full post"):
                        st.write(post_body)

                # Comments section
                comments = post.get("Comments", [])
                if comments:
                    st.markdown("**💬 Top Comments:**")
                    for j, comment in enumerate(comments[:3]):
                        comment_body = comment.get("Comment Body", "")
                        upvotes = comment.get("Upvotes", 0)
                        if len(comment_body) > 150:
                            comment_body = comment_body[:150] + "..."
                        st.markdown(f"- {comment_body} (👍 {upvotes})")
                else:
                    st.write("*No comments found*")

            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            print(f"Error displaying post {i+1}:", e)
            traceback.print_exc()
    try:
        status_text.text("Generating analysis report...")
        report_data = generate_report(posts_data)
        progress_bar.progress(100)
        status_text.text("Analysis complete")

        if report_data:
            df = pd.DataFrame(report_data)
            st.markdown("---")
            st.subheader("Analysis Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Comments Analyzed", len(df))
            with col2:
                if "Sentiment" in df.columns:
                    avg_sentiment = df["Sentiment"].mean()
                    st.metric("Avg Sentiment", f"{avg_sentiment:.2f}")
            with col3:
                if "Fact Check" in df.columns:
                    fact_check_counts = df["Fact Check"].value_counts()
                    st.metric("Unique Claims", len(fact_check_counts))
            st.write("### Detailed Results")
            st.dataframe(df, width="stretch", height=400)
            st.write("### Charts")
            col1, col2 = st.columns(2)
            with col1:
                if "Sentiment" in df.columns:
                    st.write("Sentiment Distribution")
                    sentiment_counts = df["Sentiment"].round(1).value_counts().sort_index()
                    st.line_chart(sentiment_counts)
            with col2:
                if "Fact Check" in df.columns:
                    st.write("Fact-Check Results")
                    fact_check_counts = df["Fact Check"].value_counts()
                    st.line_chart(fact_check_counts)
            st.write("### Export Data")
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV Report",
                data=csv,
                file_name="reddit_analysis_report.csv",
                mime="text/csv",
                width="stretch"
            )
        else:
            st.info("No matching comments found for analysis.")
    except Exception as e:
        st.error("Error generating report! Check terminal for details.")
        st.error(f"Error: {str(e)}")
        print("Exception in generate_report():", e)
        traceback.print_exc()



# ---------------- Floating Helper Agent ---------------- #

if "helper_expanded" not in st.session_state:
    st.session_state.helper_expanded = False
if "helper_question" not in st.session_state:
    st.session_state.helper_question = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def toggle_helper():
    st.session_state.helper_expanded = not st.session_state.helper_expanded

def handle_helper():
    question = st.session_state.helper_question.strip()
    if question:

        answer = ask_helper(question)

        st.session_state.chat_history.append({"Q": question, "A": answer})
        st.session_state.helper_question = ""



st.markdown("""
<style>
/* Floating Help Button */
.helper-btn {
  position: fixed;
  bottom: 20px;   /* distance from bottom */
  right: 20px;    /* distance from right */
  width: 60px;
  height: 60px;
  border-radius: 50%;
  background: linear-gradient(135deg, #00D4B1, #008080);
  color: #000;
  font-size: 30px;
  text-align: center;
  line-height: 60px;
  cursor: pointer;
  z-index: 9999;
  box-shadow: 0 6px 15px rgba(0,0,0,0.3);
}

/* Helper Chat Box */
.helper-box {
  position: fixed;
  bottom: 90px;  /* placed above the button */
  right: 20px;
  width: 360px;
  max-height: 450px;
  background-color: #1E1E2F;
  color: #FAFAFA;
  border-radius: 16px;
  padding: 15px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.5);
  z-index: 9998;
  overflow-y: auto;
}

.helper-box textarea {
  width: 100%;
  border-radius: 10px;
  padding: 8px;
  margin-top: 5px;
  background-color: #2A2A3D;
  color: #FAFAFA;
  border: none;
  resize: none;
}

.helper-box button {
  width: 100%;
  margin-top: 8px;
  padding: 8px;
  border-radius: 10px;
  background: linear-gradient(135deg,#00D4B1,#008080);
  color: #000;
  font-weight: bold;
  border: none;
  cursor: pointer;
}

.helper-question {
  background-color: #2A2A3D;
  border-radius: 10px;
  padding: 8px;
  margin-top: 8px;
  font-weight: bold;
}

.helper-answer {
  background-color: #262730;
  border-radius: 10px;
  padding: 8px;
  margin-top: 4px;
  font-size: 14px;
  max-height: 200px;
  overflow-y: auto;
}
</style>

""", unsafe_allow_html=True)



st.button("💡Need Help?", key="helper_toggle_btn", on_click=toggle_helper, help="Open Helper Chat")


if st.session_state.helper_expanded:
    st.markdown('<div id="helper-box">', unsafe_allow_html=True)


    st.text_area("Ask your question...", key="helper_question", height=50)
    st.button("Ask", on_click=handle_helper)


    if st.session_state.chat_history:
        for chat in st.session_state.chat_history:
            st.markdown(f'<div class="helper-question">Q: {chat["Q"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="helper-answer">A: {chat["A"]}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)






































