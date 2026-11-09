
import pandas as pd


# ==========================================
# 1. Load Dataset
# ==========================================

FILE_PATH = "dataset/processed/documents.csv"

df = pd.read_csv(FILE_PATH)


# ==========================================
# 2. Basic Information
# ==========================================

print("=" * 50)
print("DATASET INFORMATION")
print("=" * 50)

print("Number of documents:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nDataset shape:")
print(df.shape)


# ==========================================
# 3. Missing Values
# ==========================================

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)

print(df.isnull().sum())


# ==========================================
# 4. Empty Values
# ==========================================

print("\n" + "=" * 50)
print("EMPTY VALUES")
print("=" * 50)

for column in ["title", "question", "answer", "body"]:

    empty_count = (
        df[column]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print(f"{column}: {empty_count}")


# ==========================================
# 5. Duplicate Documents
# ==========================================

print("\n" + "=" * 50)
print("DUPLICATE DOCUMENTS")
print("=" * 50)

print("Duplicate document IDs:", df["document_id"].duplicated().sum())

print("Duplicate bodies:", df["body"].duplicated().sum())


# ==========================================
# 6. Document Length
# ==========================================

print("\n" + "=" * 50)
print("DOCUMENT LENGTH")
print("=" * 50)

df["body_length"] = df["body"].astype(str).apply(len)

print("Shortest document:", df["body_length"].min())

print("Longest document:", df["body_length"].max())

print("Average document length:", round(df["body_length"].mean(), 2))


# ==========================================
# 7. Display Sample Documents
# ==========================================

print("\n" + "=" * 50)
print("SAMPLE DOCUMENTS")
print("=" * 50)

for index, row in df.head(5).iterrows():

    print(f"\nDocument {index + 1}")

    print("ID:", row["document_id"])
    print("Title:", row["title"])
    print("Question:", row["question"])
    print("Answer:", row["answer"])
    print("Body:", str(row["body"])[:300], "...")


# ==========================================
# 8. Final Summary
# ==========================================

print("\n" + "=" * 50)
print("INSPECTION COMPLETED")
print("=" * 50)
