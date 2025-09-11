import logging
import time

from app import app_state, chat, setup_retriever

logging.basicConfig(level=logging.INFO)

# -------------------------
# Step 1: Define Test PDFs with 20 questions each
# -------------------------
TEST_PDFS = {
    "pride_prejudice": {
        "path": "./test_pdfs/pride_and_prejudice.pdf",
        "questions": [
            # Easy (7)
            ("Who is the main character?", "Elizabeth Bennet"),
            ("Who is Mr. Darcy?", "A wealthy gentleman and Elizabeth’s love interest"),
            ("How many Bennet sisters are there?", "Five"),
            ("Who marries Mr. Bingley?", "Jane Bennet"),
            ("Who is Lydia Bennet?", "The youngest Bennet sister"),
            ("Who is Mr. Wickham?", "A charming officer who deceives Elizabeth"),
            ("What estate does Mr. Darcy own?", "Pemberley"),
            # Medium (8)
            (
                "What happens at the Meryton ball?",
                "Elizabeth meets Mr. Darcy; initial tension occurs",
            ),
            (
                "Why does Mr. Darcy initially dislike Elizabeth?",
                "He thinks she is beneath his social class",
            ),
            ("Who is Charlotte Lucas?", "Elizabeth’s friend who marries Mr. Collins"),
            ("Why does Elizabeth reject Mr. Collins?", "She does not love him"),
            ("Who helps Lydia after she elopes?", "Mr. Darcy"),
            ("Who is Lady Catherine de Bourgh?", "Mr. Darcy’s wealthy, meddling aunt"),
            (
                "How does Jane feel about Bingley?",
                "She loves him but is shy and reserved",
            ),
            (
                "What is the main theme of the novel?",
                "Pride and prejudice affecting relationships",
            ),
            # Hard (5)
            (
                "Explain Mr. Darcy's character development.",
                "He becomes more humble and loving, overcoming pride",
            ),
            (
                "Why does Elizabeth initially misjudge Darcy?",
                "She is influenced by Wickham’s lies and her own prejudices",
            ),
            (
                "How does Mr. Darcy help the Bennet family?",
                "He arranges Lydia’s marriage to Wickham",
            ),
            (
                "What role does social class play in the story?",
                "It affects marriage prospects and personal judgments",
            ),
            (
                "Summarize the ending in one sentence.",
                "Elizabeth and Darcy marry, and the family is reconciled",
            ),
        ],
    }
}


# -------------------------
# Step 2: Test Function
# -------------------------
def test_pdf(pdf_name, pdf_info):
    print(f"\n=== Testing PDF: {pdf_name} ===")
    pdf_path = pdf_info["path"]

    # Setup retriever
    print("Setting up retriever...")
    retriever = setup_retriever(pdf_path)
    app_state["retriever"] = retriever

    total_questions = len(pdf_info["questions"])
    correct = 0
    response_times = []

    for question, expected_answer in pdf_info["questions"]:
        print(f"\nQuestion: {question}")
        start = time.perf_counter()

        # Generate response using generator
        response_text = ""
        for partial in chat(question, []):
            response_text = partial  # take the latest streamed output

        end = time.perf_counter()
        response_times.append(end - start)

        print("Response:", response_text)

    avg_time = sum(response_times) / len(response_times)

    print(f"Average Response Time: {avg_time:.2f} seconds")


# -------------------------
# Step 3: Run All Tests
# -------------------------
if __name__ == "__main__":
    for pdf_name, pdf_info in TEST_PDFS.items():
        test_pdf(pdf_name, pdf_info)
