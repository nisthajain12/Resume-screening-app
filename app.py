# app.py

import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import docx2txt
import pickle
import re
import altair as alt

# -------------------------------
# 1. Load Model, Vectorizer, Role-Skills
# -------------------------------
with open("src/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("src/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("src/role_skills.pkl", "rb") as f:
    role_skills = pickle.load(f)

all_roles = list(role_skills.keys())

# -------------------------------
# 2. Resume Text Extraction
# -------------------------------
def extract_text(file):
    if file.type == "application/pdf":
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in pdf:
            text += page.get_text()
        return text
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(file)
    elif file.type == "text/plain":
        return file.read().decode("utf-8")
    else:
        return ""

# -------------------------------
# 3. Clean Text
# -------------------------------
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -------------------------------
# 4. Skill Extraction
# -------------------------------
def extract_skills(text):
    skills_found = []
    for role, skills in role_skills.items():
        for skill in skills:
            if skill.lower() in text:
                skills_found.append(skill.lower())
    return list(set(skills_found))

# -------------------------------
# 5. Role Prediction Based on Skills
# -------------------------------
def predict_role(skills_found):
    role_match_count = {}
    for role, skills in role_skills.items():
        match_count = len(set(skills_found) & set(skills))
        role_match_count[role] = match_count

    # Sort roles by number of matched skills
    sorted_roles = sorted(role_match_count.items(), key=lambda x: x[1], reverse=True)
    best_role = sorted_roles[0][0]
    secondary_roles = [role for role, count in sorted_roles[1:3]]  # top 2 secondary roles

    # Skills to master for best role
    best_role_skills_to_master = list(set(role_skills[best_role]) - set(skills_found))

    # Skills to master for secondary roles
    secondary_roles_suggestions = {}
    for role in secondary_roles:
        missing_skills = list(set(role_skills[role]) - set(skills_found))
        secondary_roles_suggestions[role] = missing_skills

    return best_role, best_role_skills_to_master, secondary_roles_suggestions

# -------------------------------
# 6. Streamlit UI
# -------------------------------
st.set_page_config(page_title="Resume Screening App", page_icon="📄", layout="centered")
st.title("📄 Resume Screening App")
st.write("Upload a resume to see its predicted role and skill suggestions.")

uploaded_file = st.file_uploader("Upload Resume (PDF, DOCX, TXT)", type=["pdf","docx","txt"])

if uploaded_file:
    with st.spinner("Extracting text from resume..."):
        resume_text = extract_text(uploaded_file)
        cleaned_text = clean_text(resume_text)

    with st.spinner("Extracting skills..."):
        skills_found = extract_skills(cleaned_text)

    with st.spinner("Predicting best role..."):
        best_role, best_role_skills_to_master, secondary_roles_suggestions = predict_role(skills_found)
        # -------------------------------
    # ATS Friendliness Checker (before role prediction)
    # -------------------------------
    def ats_friendliness_score(text, skills_found):
        score = 0
        max_score = 100

        # 1. Presence of relevant skills in resume
        total_skills = sum(len(skills) for skills in role_skills.values())
        if total_skills > 0:
            skill_coverage = len(skills_found) / total_skills
            score += int(skill_coverage * 50)  # 50 points for skills coverage

        # 2. Resume length check
        if len(text) < 1000:
            score += 10  # shorter resumes less preferred
        else:
            score += 20

        # 3. Readability: penalize excessive special characters
        non_alpha_ratio = len(re.findall(r"[^a-zA-Z\s]", text)) / max(len(text), 1)
        if non_alpha_ratio < 0.05:
            score += 20
        elif non_alpha_ratio < 0.1:
            score += 10
        else:
            score += 0

        return min(score, max_score)

    ats_score = ats_friendliness_score(cleaned_text, skills_found)
    st.subheader("📄 ATS Friendliness Score")
    st.progress(ats_score / 100)
    st.write(f"Your resume is **{ats_score}% ATS-friendly**.")
    st.info(
        "A higher score means your resume is more likely to be parsed correctly by Applicant Tracking Systems (ATS)."
    )


    # Compute Role Match Percentages
    role_match_percent = {}
    for role, skills in role_skills.items():
        match_count = len(set(skills_found) & set(skills))
        role_match_percent[role] = (match_count / len(skills)) * 100  # % match

    df_roles = pd.DataFrame(role_match_percent.items(), columns=['Role', 'Match %'])
    df_roles = df_roles.sort_values(by='Match %', ascending=False).head(6)

    # Display bar chart
    st.subheader("📊 Role Match Percentage")
    chart = alt.Chart(df_roles).mark_bar().encode(
        x='Match %:Q',
        y=alt.Y('Role:N', sort='-x'),
        tooltip=['Role', 'Match %']
    ).properties(title='Top Role Matches (%)')
    st.altair_chart(chart, use_container_width=True)

    # Display Predicted Role
    st.subheader("✅ Predicted Primary Role")
    st.success(f"{best_role}")

    # Display Skills Found
    st.subheader("🛠 Skills Present in Resume")
    if skills_found:
        skills_df = pd.DataFrame([skills_found[i:i+5] for i in range(0, len(skills_found), 5)])
        st.table(skills_df.fillna(""))
    else:
        st.write("No predefined skills found in resume.")

    # Display Skills to Master
    st.subheader(f"💡 Skills to Master for {best_role}")
    if best_role_skills_to_master:
        master_df = pd.DataFrame([best_role_skills_to_master[i:i+5] for i in range(0, len(best_role_skills_to_master), 5)])
        st.table(master_df.fillna(""))
    else:
        st.write("You already have all core skills for this role!")

    # Display Secondary Roles
    st.subheader("🎯 Secondary Roles You Could Target")
    for role, missing_skills in secondary_roles_suggestions.items():
        st.markdown(f"**{role}:**")
        if missing_skills:
            sec_skills_df = pd.DataFrame([missing_skills[i:i+5] for i in range(0, len(missing_skills), 5)])
            st.table(sec_skills_df.fillna(""))
        else:
            st.write("You already have most of the skills for this role!")

    # -------------------------------
    # Learning Resources
    st.subheader("📚 Learning Resources for Skills to Master")
    skill_resources = {
    "python": "https://www.learnpython.org/",
    "django": "https://www.djangoproject.com/start/",
    "flask": "https://flask.palletsprojects.com/en/2.3.x/tutorial/",
    "aws": "https://aws.amazon.com/training/",
    "docker": "https://www.docker.com/101-tutorial",
    "kubernetes": "https://kubernetes.io/docs/tutorials/",
    "sql": "https://www.w3schools.com/sql/",
    "excel": "https://www.coursera.org/learn/excel-data-analysis",
    "javascript": "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide",
    "react": "https://reactjs.org/tutorial/tutorial.html",
    "java": "https://www.learnjavaonline.org/",
    "spring": "https://spring.io/guides",
    "selenium": "https://www.selenium.dev/documentation/",
    # --- New additions ---
    "cpp": "https://www.learncpp.com/",
    "exploitation": "https://owasp.org/",  # OWASP: ethical web-security guidance and safe learning resources
    "firewall": "https://www.pfsense.org/learn/",
    "api": "https://swagger.io/docs/",
    "angular": "https://angular.io/tutorial",
    "css": "https://developer.mozilla.org/en-US/docs/Web/CSS",
    "html": "https://developer.mozilla.org/en-US/docs/Web/HTML",
    "linux": "https://linuxjourney.com/",
    "automation": "https://www.ansible.com/resources/get-started",
    "bash": "https://linuxcommand.org/",
    "git": "https://git-scm.com/docs/gittutorial",
    "nodejs": "https://nodejs.dev/learn",
    "typescript": "https://www.typescriptlang.org/docs/handbook/intro.html",
    "graphql": "https://graphql.org/learn/",
    "rest": "https://restfulapi.net/",
    "ci_cd": "https://www.jenkins.io/doc/tutorials/",
    "security": "https://www.mitre.org/"  # MITRE for ATT&CK etc. (defensive/security frameworks)
}

    if best_role_skills_to_master:
        for i in range(0, len(best_role_skills_to_master), 3):
            cols = st.columns(3)
            for j, skill in enumerate(best_role_skills_to_master[i:i+3]):
                url = skill_resources.get(skill, "#")
                with cols[j]:
                    st.markdown(
    f'''
    <a href="{url}" target="_blank">
        <button style="
            width: 100%;
            background-color: #1f1f1f;  /* dark color */
            color: white;               /* text color */
            border: none;
            padding: 8px;
            border-radius: 5px;
            font-weight: bold;
            cursor: pointer;
        ">{skill.capitalize()}</button>
    </a>
    ''',
    unsafe_allow_html=True
)

    else:
        st.write("No additional learning resources needed; you already have all core skills!")

    st.markdown("---")
    st.info("Skills are matched based on a predefined role-skills dictionary derived from Gaurav Dutta's Kaggle dataset.")


# -------------------------------
# Top LeetCode Questions by Role
# -------------------------------
role_leetcode_questions = {
    "Data Science": {
        "Two Sum": "https://leetcode.com/problems/two-sum/",
        "Median of Two Sorted Arrays": "https://leetcode.com/problems/median-of-two-sorted-arrays/",
        "Kth Largest Element in an Array": "https://leetcode.com/problems/kth-largest-element-in-an-array/"
    },
    "HR": {
        "Valid Parentheses": "https://leetcode.com/problems/valid-parentheses/",
        "Group Anagrams": "https://leetcode.com/problems/group-anagrams/",
        "Meeting Rooms II": "https://leetcode.com/problems/meeting-rooms-ii/"
    },
    "Arts Teacher": {
        "Flood Fill": "https://leetcode.com/problems/flood-fill/",
        "Unique Paths": "https://leetcode.com/problems/unique-paths/",
        "Generate Parentheses": "https://leetcode.com/problems/generate-parentheses/"
    },
    "Web Designer": {
        "Valid Palindrome": "https://leetcode.com/problems/valid-palindrome/",
        "Minimum Window Substring": "https://leetcode.com/problems/minimum-window-substring/",
        "Longest Common Prefix": "https://leetcode.com/problems/longest-common-prefix/"
    },
    "Mechanical Engineer": {
        "Design HashMap": "https://leetcode.com/problems/design-hashmap/",
        "Rotate Image": "https://leetcode.com/problems/rotate-image/",
        "Trapping Rain Water": "https://leetcode.com/problems/trapping-rain-water/"
    },
    "Sales": {
        "Best Time to Buy and Sell Stock": "https://leetcode.com/problems/best-time-to-buy-and-sell-stock/",
        "Task Scheduler": "https://leetcode.com/problems/task-scheduler/",
        "Top K Frequent Elements": "https://leetcode.com/problems/top-k-frequent-elements/"
    },
    "Health and Fitness Trainer": {
        "Climbing Stairs": "https://leetcode.com/problems/climbing-stairs/",
        "Maximum Subarray": "https://leetcode.com/problems/maximum-subarray/",
        "Longest Increasing Subsequence": "https://leetcode.com/problems/longest-increasing-subsequence/"
    },
    "Civil Engineer": {
        "Max Area of Island": "https://leetcode.com/problems/max-area-of-island/",
        "Walls and Gates": "https://leetcode.com/problems/walls-and-gates/",
        "Course Schedule": "https://leetcode.com/problems/course-schedule/"
    },
    "Java Developer": {
        "Implement strStr()": "https://leetcode.com/problems/implement-strstr/",
        "Reverse Integer": "https://leetcode.com/problems/reverse-integer/",
        "Valid Parentheses": "https://leetcode.com/problems/valid-parentheses/"
    },
    "Python Developer": {
        "Add Two Numbers": "https://leetcode.com/problems/add-two-numbers/",
        "Valid Anagram": "https://leetcode.com/problems/valid-anagram/",
        "Word Break": "https://leetcode.com/problems/word-break/"
    },
    "Full Stack Developer": {
        "Design Twitter": "https://leetcode.com/problems/design-twitter/",
        "LFU Cache": "https://leetcode.com/problems/lfu-cache/",
        "Serialize and Deserialize Binary Tree": "https://leetcode.com/problems/serialize-and-deserialize-binary-tree/"
    },
    "Frontend Developer": {
        "Design Browser History": "https://leetcode.com/problems/design-browser-history/",
        "Number of Islands": "https://leetcode.com/problems/number-of-islands/",
        "Longest Increasing Subsequence": "https://leetcode.com/problems/longest-increasing-subsequence/"
    },
    "Database Engineer": {
        "Database Queries (SQL I)": "https://leetcode.com/studyplan/sql/",
        "Employees Earning More Than Their Managers": "https://leetcode.com/problems/employees-earning-more-than-their-managers/",
        "Department Highest Salary": "https://leetcode.com/problems/department-highest-salary/"
    },
    "DevOps Engineer": {
        "Min Stack": "https://leetcode.com/problems/min-stack/",
        "Evaluate Reverse Polish Notation": "https://leetcode.com/problems/evaluate-reverse-polish-notation/",
        "Design Circular Queue": "https://leetcode.com/problems/design-circular-queue/"
    },
    "Network Security Engineer": {
        "Network Delay Time": "https://leetcode.com/problems/network-delay-time/",
        "Evaluate Division": "https://leetcode.com/problems/evaluate-division/",
        "Redundant Connection": "https://leetcode.com/problems/redundant-connection/"
    },
    "Ethical Hacker": {
        "Keys and Rooms": "https://leetcode.com/problems/keys-and-rooms/",
        "Word Ladder": "https://leetcode.com/problems/word-ladder/",
        "Open the Lock": "https://leetcode.com/problems/open-the-lock/"
    },
    "Business Analyst": {
        "Range Sum Query - Immutable": "https://leetcode.com/problems/range-sum-query-immutable/",
        "Product of Array Except Self": "https://leetcode.com/problems/product-of-array-except-self/",
        "Pivot Index": "https://leetcode.com/problems/find-pivot-index/"
    },
    "Automation Tester": {
        "Implement Queue using Stacks": "https://leetcode.com/problems/implement-queue-using-stacks/",
        "String to Integer (atoi)": "https://leetcode.com/problems/string-to-integer-atoi/",
        "Valid Sudoku": "https://leetcode.com/problems/valid-sudoku/"
    },
    "PMO": {
        "Meeting Rooms II": "https://leetcode.com/problems/meeting-rooms-ii/",
        "Course Schedule II": "https://leetcode.com/problems/course-schedule-ii/",
        "Task Scheduler": "https://leetcode.com/problems/task-scheduler/"
    },
    "Blockchain Developer": {
        "Valid Blockchain Transactions (custom-like)": "https://leetcode.com/problems/valid-parentheses/",
        "Encode and Decode Strings": "https://leetcode.com/problems/encode-and-decode-strings/",
        "Find Duplicate Subtrees": "https://leetcode.com/problems/find-duplicate-subtrees/"
    },
    "ETL Developer": {
        "Data Stream as Disjoint Intervals": "https://leetcode.com/problems/data-stream-as-disjoint-intervals/",
        "Intersection of Two Arrays II": "https://leetcode.com/problems/intersection-of-two-arrays-ii/",
        "Find All Anagrams in a String": "https://leetcode.com/problems/find-all-anagrams-in-a-string/"
    },
    "SAP Developer": {
        "Merge Intervals": "https://leetcode.com/problems/merge-intervals/",
        "Insert Interval": "https://leetcode.com/problems/insert-interval/",
        "Minimum Path Sum": "https://leetcode.com/problems/minimum-path-sum/"
    }
}
if uploaded_file and best_role:
    st.subheader(f"💻 Top LeetCode Questions for {best_role}")

    questions = role_leetcode_questions.get(best_role, {})

    if questions:
        for i, (q_name, q_link) in enumerate(questions.items()):
            st.markdown(
                f'<a href="{q_link}" target="_blank">'
                f'<button style="width:100%; background-color:#1f1f1f; color:white; border:none; padding:8px; border-radius:5px; margin-bottom:5px;">{q_name}</button>'
                f'</a>',
                unsafe_allow_html=True
            )
    else:
        st.write("No curated LeetCode questions available for this role yet.")

