import os
import time
import traceback
import praw
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv

# ---------------- Load environment variables ---------------- #
load_dotenv("api.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
REDDIT_USERNAME = os.getenv("REDDIT_USERNAME")
REDDIT_PASSWORD = os.getenv("REDDIT_PASSWORD")

if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT, REDDIT_USERNAME, REDDIT_PASSWORD]):
    raise ValueError("Please set Reddit API credentials (including username & password) in your .env file!")

# ---------------- Initialize PRAW for posting ---------------- #
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT,
    username=REDDIT_USERNAME,
    password=REDDIT_PASSWORD
)

# ---------------- Agents ---------------- #
collector_agent = Agent(
    role="Reddit Data Collector & Summarizer",
    goal="Collect top posts and summarize discussions with structured output",
    backstory="An expert Reddit analyst who reads threads and summarizes their tone, content, and top comments clearly.",
    llm="gpt-4o-mini",
)

sentiment_agent = Agent(
    role="Sentiment Analysis Agent",
    goal="Return a numeric sentiment score between -1.0 and +1.0",
    backstory="A sentiment scoring AI that only outputs a single number.",
    llm="gpt-4o-mini",
)

factchecker_agent = Agent(
    role="Fact-Checking Agent",
    goal="Verify factual accuracy of comments and output True, False, or Unverified",
    backstory="A concise fact-checking AI that only outputs one word.",
    llm="gpt-4o-mini",
)

comment_agent = Agent(
    role="Reddit Comment Generator",
    goal="Given a top comment, generate a new one with similar tone but fresh perspective",
    backstory="An AI that mimics human Redditors' tone and insight in comments.",
    llm="gpt-4o-mini",
)

helper_agent = Agent(
    role="Helper",
    goal="Explain and resolve project-related doubts in simple terms",
    backstory=(
        "A friendly AI tutor that explains the functioning of agents, Reddit posting, and fetching logic. "
        "Provides solutions, examples, and fixes clearly."
    ),
    llm="gpt-4o-mini",
)

# ---------------- Helper Function ---------------- #
def ask_helper(question: str):
    try:
        task = Task(
            description=question,
            agent=helper_agent,
            expected_output="A clear, beginner-friendly explanation relevant to the Reddit Insight Agent project."
        )
        crew = Crew(agents=[helper_agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        return result.raw if hasattr(result, 'raw') else str(result)
    except Exception as e:
        print("❌ Error in Helper Agent:", e)
        return "Sorry, Helper Agent could not answer."

# ---------------- Comment Generator ---------------- #
def generate_comment_from_best(fetched_comments):
    if not fetched_comments:
        return None

    best_comment = max(fetched_comments, key=lambda c: c.get('Upvotes', 0))
    prompt = f"Original Comment: {best_comment['Comment Body']}\nGenerate a new comment in a similar style."

    try:
        task = Task(
            description=prompt,
            agent=comment_agent,
            expected_output="A natural Reddit-style comment written in similar tone and context."
        )
        crew = Crew(agents=[comment_agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        return result.raw.strip() if hasattr(result, 'raw') else str(result).strip()
    except Exception as e:
        print("Error generating comment:", e)
        return None

# ---------------- Fetch Reddit Posts ---------------- #
def fetch_posts(subreddits, keywords=None, post_limit=None, comment_limit=None):
    raw_data = []
    try:
        post_limit = post_limit or 2
        comment_limit = comment_limit or 3

        for subreddit in subreddits:
            top_posts = reddit.subreddit(subreddit).top(limit=post_limit)
            for post in top_posts:
                post.comments.replace_more(limit=0)
                all_comments = post.comments.list()

                filtered_comments = [
                    {"Comment Body": c.body, "Upvotes": getattr(c, "score", 0)}
                    for c in all_comments
                    if not keywords or any(kw.lower() in c.body.lower() for kw in keywords)
                ]

                sorted_comments = sorted(filtered_comments, key=lambda x: x["Upvotes"], reverse=True)
                post_comments = sorted_comments[:comment_limit]

                print(f"DEBUG: Post: {post.title}, Top Comments Fetched: {len(post_comments)}")

                try:
                    comments_text = "\n".join(
                        [f"{i+1}. {c['Comment Body']}" for i, c in enumerate(post_comments)]
                    )
                    prompt = (
                        f"Subreddit: {subreddit}\nPost: {post.title}\nComments:\n{comments_text}\n\n"
                        "For each post, generate exactly:\n"
                        "Post Title: <Post title>\nCollector Summary: <3-5 sentence summary>\n"
                        "Comments:\n1. Comment Body: <first comment>\n   Upvotes: <number>\n"
                        "2. Comment Body: <second comment>\n   Upvotes: <number>\n"
                        "Overall Discussion Tone: <Neutral / Supportive / Critical / Mixed>"
                    )
                    task = Task(
                        description=prompt,
                        agent=collector_agent,
                        expected_output="A structured summary with post title, summary, top comments, and tone classification."
                    )
                    crew = Crew(agents=[collector_agent], tasks=[task], process=Process.sequential)
                    result = crew.kickoff()
                    collector_text = result.raw.strip() if hasattr(result, 'raw') else str(result).strip()
                except Exception:
                    collector_text = "Collector agent failed"

                raw_data.append({
                    "Subreddit": subreddit,
                    "Post Title": post.title,
                    "Post Body": getattr(post, "selftext", "") or "",
                    "Post Link": f"https://reddit.com{post.permalink}",
                    "Post Upvotes": getattr(post, "score", 0),
                    "Post Thumbnail": post.thumbnail if hasattr(post, "thumbnail") and post.thumbnail.startswith("http") else None,
                    "Collector Summary": collector_text,
                    "Comments": post_comments
                })
                time.sleep(1)

        print(f"DEBUG: Total Posts Fetched: {len(raw_data)}")
        return raw_data
    except Exception as e:
        print("Exception in fetch_posts():", e)
        traceback.print_exc()
        return []

# ---------------- Generate Report ---------------- #
def generate_report(posts_data):
    report = []
    for post in posts_data:
        for comment in post.get("Comments", []):
            body = comment.get("Comment Body", "")
            sentiment_score = 0.0
            verdict = "Unverified"

            try:
                task = Task(
                    description=f"Analyze sentiment (positive=1, neutral=0, negative=-1). Comment: {body}",
                    agent=sentiment_agent,
                    expected_output="A single numeric sentiment score between -1.0 and +1.0"
                )
                crew = Crew(agents=[sentiment_agent], tasks=[task], process=Process.sequential)
                result = crew.kickoff()
                sentiment_score = float(result.raw.strip()) if hasattr(result, 'raw') else float(str(result).strip())
            except Exception as e:
                print("Error in sentiment analysis:", e)
                traceback.print_exc()

            try:
                task = Task(
                    description=f"Fact check this comment. Respond only with True, False, or Unverified:\n{body}",
                    agent=factchecker_agent,
                    expected_output="One of: True, False, or Unverified"
                )
                crew = Crew(agents=[factchecker_agent], tasks=[task], process=Process.sequential)
                result = crew.kickoff()
                verdict = result.raw.strip() if hasattr(result, 'raw') else str(result).strip()
            except Exception as e:
                print("Error in fact checking:", e)
                traceback.print_exc()

            report.append({
                "Subreddit": post.get("Subreddit"),
                "Post Title": post.get("Post Title"),
                "Post Link": post.get("Post Link"),
                "Post Upvotes": post.get("Post Upvotes", 0),
                "Collector Summary": post.get("Collector Summary", ""),
                "Comment": body,
                "Comment Upvotes": comment.get("Upvotes", 0),
                "Sentiment": sentiment_score,
                "Fact Verdict": verdict
            })
    return report

# ---------------- Create Reddit Post ---------------- #
def create_post(subreddit, title, body, flair_text=None):
    try:
        subreddit_obj = reddit.subreddit(subreddit)

        if flair_text:
            flair_id = None
            for ft in subreddit_obj.flair.link_templates:
                if ft['text'].lower() == flair_text.lower():
                    flair_id = ft['id']
                    break
            submission = subreddit_obj.submit(title=title, selftext=body, flair_id=flair_id)
        else:
            submission = subreddit_obj.submit(title=title, selftext=body)

        print(f"✅ Post created: https://reddit.com{submission.permalink}")
        return submission
    except Exception as e:
        print("❌ Error while posting:", e)
        return None
