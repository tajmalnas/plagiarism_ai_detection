import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import crawl  
from ai_detect import generate_ai_essay, check_ai_similarity  
import os
from grading import grade_essay  # Importing the grading module

def calculate_tfidf_similarity(essay, docs):
    """
    Compute TF-IDF similarity between the given essay and fetched documents.
    """
    all_texts = [essay] + [doc["content"] for doc in docs]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Compute cosine similarity between the essay and all documents
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Pair similarity scores with sources
    similarity_results = [(docs[i]["source"], similarities[i]) for i in range(len(docs))]
    
    return sorted(similarity_results, key=lambda x: x[1], reverse=True)

def save_ai_essays(topic, ai_essays):
    """
    Saves AI-generated essays to a CSV file.
    """
    df = pd.DataFrame({"Topic": topic, "AI_Generated_Essay": ai_essays})
    file_exists = os.path.isfile("ai_generated_essays.csv")

    # Append if file exists, otherwise create new
    df.to_csv("ai_generated_essays.csv", mode="a", header=not file_exists, index=False, encoding="utf-8")

def main():
    st.title("Plagiarism, AI Detection & Essay Grading")
    topic = st.text_input("Enter the Topic:")
    essay_text = st.text_area("Paste your Essay:")
    
    if st.button("Check Essay"):
        if topic and essay_text:
            st.write("### Fetching documents related to topic:", topic)
            docs = crawl.get_docs(topic, num_results=10)
            
            st.write("### Calculating similarity scores...")
            similarity_results = calculate_tfidf_similarity(essay_text, docs)
            
            st.write("### Plagiarism Report:")
            df = pd.DataFrame(similarity_results, columns=["Source", "Similarity Score"])
            st.dataframe(df)
            
            # Save plagiarism results
            df.to_csv("plagiarism_results.csv", index=False, encoding="utf-8")
            st.success("Plagiarism report saved as plagiarism_results.csv")

            # AI Detection
            st.write("### AI Detection Analysis:")
            st.write("Generating AI-written essays for comparison...")
            ai_essays = generate_ai_essay(topic, num_variations=10)
            
            # Save AI essays to CSV
            save_ai_essays(topic, ai_essays)

            # Check AI similarity
            ai_similarity_score = check_ai_similarity(essay_text, ai_essays)
            st.write(f"**AI Similarity Score:** {ai_similarity_score:.2f}")

            # Display AI-generated essays with similarity
            ai_similarities = [
                (ai_essays[i], cosine_similarity(
                    TfidfVectorizer().fit_transform([essay_text, ai_essays[i]])[0:1],
                    TfidfVectorizer().fit_transform([essay_text, ai_essays[i]])[1:]).flatten()[0]
                ) for i in range(len(ai_essays))
            ]

            ai_df = pd.DataFrame(ai_similarities, columns=["AI Generated Essay", "Similarity Score"])
            st.write("### AI-Generated Essays & Similarity:")
            st.dataframe(ai_df)

            # AI Similarity Alert
            if ai_similarity_score > 0.8:
                st.error("⚠️ High similarity with AI-generated content! Possible AI-written essay.")
            elif ai_similarity_score > 0.5:
                st.warning("⚠️ Medium similarity with AI-generated essays. Check for plagiarism.")
            else:
                st.success("✅ Low similarity with AI-generated essays. Likely original content.")

            # **Essay Grading with Detailed Feedback**
            if topic and essay_text:
             st.write("### 📊 Evaluating Essay Quality...")
             # Call the grading function
             final_score, feedback = grade_essay(topic, essay_text)
             # Display final score with a progress bar
             st.subheader(f"🏆 Final Score: **{final_score*6/10} / 10**")
             st.progress(final_score / 10)  # Convert to a fraction of 10
             # Loop through each criterion and display it in an expandable section
             for criterion, details in feedback.items():
                 with st.expander(f"📌 {criterion} (Score: {details['Score']}/100)"):
                     st.markdown(f"**📝 Justification:** {details['Justification']}")
                     st.success(f"✅ **Strengths:** {details['Strengths']}")
                     st.warning(f"⚠️ **Areas for Improvement:** {details['Areas for Improvement']}")

        else:
            st.warning("⚠️ Please enter both the topic and the essay text.")

if __name__ == "__main__":
    main()
