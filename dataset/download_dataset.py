
import os
import re
import pandas as pd


# ==========================================
# 1. Project Configuration
# ==========================================

DATASET_URL = (
    "https://huggingface.co/datasets/rajpurkar/squad/"
    "resolve/refs%2Fconvert%2Fparquet/plain_text/train/0000.parquet"
)

OUTPUT_PATH = "dataset/processed/documents.csv"

NUMBER_OF_DOCUMENTS = 100
RANDOM_STATE = 42


# Educational keywords
EDUCATIONAL_KEYWORDS = [
    "university",
    "college",
    "education",
    "student",
    "school",
    "computer",
    "science",
    "technology",
    "database",
    "programming",
    "software",
    "engineering",
    "mathematics",
    "research",
    "learning",
    "artificial intelligence",
    "network",
    "information",
    "algorithm",
    "system"
]


# ==========================================
# 2. Text Normalization
# ==========================================

def normalize_text(text):

    """
    Normalize text to help identify duplicate documents.
    """

    text = str(text).strip().lower()

    text = re.sub(r"\s+", " ", text)

    return text


# ==========================================
# 3. Load Dataset
# ==========================================

def load_squad_dataset():

    print("Loading SQuAD dataset...")

    df = pd.read_parquet(DATASET_URL)

    print(f"Total original records: {len(df)}")

    print("\nAvailable columns:")
    print(df.columns.tolist())

    return df


# ==========================================
# 4. Select Required Columns
# ==========================================

def select_columns(df):

    print("\nSelecting required columns...")

    selected_df = df[
        ["title", "question", "context", "answers"]
    ].copy()

    # Extract the first answer
    def extract_answer(answers):

        if isinstance(answers, dict):

            answer_list = answers.get("text", [])

            if len(answer_list) > 0:

                return str(answer_list[0]).strip()

        return ""

    selected_df["answer"] = selected_df["answers"].apply(
        extract_answer
    )

    # Rename context to body
    selected_df = selected_df.rename(
        columns={"context": "body"}
    )

    # Keep only the required columns
    selected_df = selected_df[
        ["title", "question", "answer", "body"]
    ]

    return selected_df


# ==========================================
# 5. Clean Missing and Empty Values
# ==========================================

def clean_data(df):

    print("\nCleaning missing and empty values...")

    initial_count = len(df)

    # Remove missing values
    df = df.dropna(
        subset=["title", "question", "answer", "body"]
    ).copy()

    # Remove leading and trailing spaces
    for column in ["title", "question", "answer", "body"]:

        df[column] = df[column].astype(str).str.strip()

    # Remove empty records
    for column in ["title", "question", "answer", "body"]:

        df = df[df[column] != ""]

    removed_count = initial_count - len(df)

    print(f"Removed records: {removed_count}")
    print(f"Records after cleaning: {len(df)}")

    return df.reset_index(drop=True)


# ==========================================
# 6. Remove Duplicate Documents
# ==========================================

def remove_duplicates(df):

    print("\nRemoving duplicate documents...")

    initial_count = len(df)

    # Normalize body before checking duplicates
    df["normalized_body"] = df["body"].apply(
        normalize_text
    )

    # Remove duplicate bodies
    df = df.drop_duplicates(
        subset=["normalized_body"]
    ).copy()

    # Remove temporary column
    df = df.drop(columns=["normalized_body"])

    removed_count = initial_count - len(df)

    print(f"Duplicate records removed: {removed_count}")

    return df.reset_index(drop=True)


# ==========================================
# 7. Filter Educational Documents
# ==========================================

def filter_educational_documents(df):

    print("\nFiltering educational documents...")

    # Build regular expression
    pattern = (
        r"\b(?:"
        + "|".join(
            re.escape(word)
            for word in EDUCATIONAL_KEYWORDS
        )
        + r")\b"
    )

    # Search in title or body
    mask = (
        df["title"].str.contains(
            pattern,
            case=False,
            regex=True,
            na=False
        )
        |
        df["body"].str.contains(
            pattern,
            case=False,
            regex=True,
            na=False
        )
    )

    filtered_df = df[mask].copy()

    print(f"Educational records found: {len(filtered_df)}")

    return filtered_df.reset_index(drop=True)


# ==========================================
# 8. Select Sample
# ==========================================

def select_sample(df):

    print("\nSelecting sample documents...")

    if len(df) < NUMBER_OF_DOCUMENTS:

        raise ValueError(
            f"Only {len(df)} records were found. "
            f"We need at least {NUMBER_OF_DOCUMENTS}."
        )

    sampled_df = df.sample(
        n=NUMBER_OF_DOCUMENTS,
        random_state=RANDOM_STATE
    ).copy()

    print(f"Selected records: {len(sampled_df)}")

    return sampled_df.reset_index(drop=True)


# ==========================================
# 9. Generate Unique Document IDs
# ==========================================

def generate_document_ids(df):

    print("\nGenerating document IDs...")

    df.insert(
        0,
        "document_id",
        [
            f"UKS{i:04d}"
            for i in range(1, len(df) + 1)
        ]
    )

    return df


# ==========================================
# 10. Save CSV File
# ==========================================

def save_dataset(df):

    print("\nSaving dataset...")

    # Create output directory if it does not exist
    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Dataset saved successfully: {OUTPUT_PATH}")


# ==========================================
# 11. Main Program
# ==========================================

def main():

    print("=" * 55)
    print("University Knowledge Search - Dataset Preparation")
    print("=" * 55)

    # Step 1: Load dataset
    df = load_squad_dataset()

    # Step 2: Select columns and extract answers
    df = select_columns(df)

    # Step 3: Clean data
    df = clean_data(df)

    # Step 4: Remove duplicate documents
    df = remove_duplicates(df)

    # Step 5: Filter educational documents
    df = filter_educational_documents(df)

    # Step 6: Select 100 records
    df = select_sample(df)

    # Step 7: Generate unique IDs
    df = generate_document_ids(df)

    # Step 8: Save CSV
    save_dataset(df)

    print("\n" + "=" * 55)
    print("DATASET PREPARATION COMPLETED")
    print("=" * 55)

    print(f"\nFinal number of documents: {len(df)}")

    print("\nFinal columns:")
    print(df.columns.tolist())

    print("\nFirst 5 records:")

    print(
        df[
            ["document_id", "title", "question", "answer"]
        ].head()
    )


# ==========================================
# 12. Program Entry Point
# ==========================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print("\nERROR OCCURRED:")
        print(error)
