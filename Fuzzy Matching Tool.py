import pandas as pd
import numpy as np
import os
import logging
import pickle
import faiss
import glob
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from pathlib import Path
from datetime import datetime
import time

# -------------------------------------------------------------------
# Configuration & Logging
# -------------------------------------------------------------------
def setup_logging(log_file="fuzzy_lookup.log"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

# -------------------------------------------------------------------
# User Input Functions
# -------------------------------------------------------------------
def get_api_key():
    """Get OpenAI API key: check environment variable first, then prompt."""
    api_key = os.getenv('OPENAI_API_KEY', '').strip()
    if api_key:
        print("Using OpenAI API key from OPENAI_API_KEY environment variable.")
        return api_key

    print("\n=== OpenAI API Key ===")
    print("No OPENAI_API_KEY environment variable found.")
    while True:
        api_key = input("Please enter your OpenAI API key: ").strip()
        if api_key:
            return api_key
        print("API key cannot be empty. Please try again.")


def get_file_or_folder(label):
    """File/folder selection with clear labeling."""
    print(f"\nWhat type of file is your {label}?")
    print("  1. Excel file (.xlsx / .xls)")
    print("  2. CSV file (.csv)")
    print("  3. Folder of CSV files")

    choice = input("\nEnter your choice (1-3): ").strip()

    if choice == "1":
        path = input(f"\nDrag and drop your {label} Excel file here (or type the path): ").strip().strip('"').strip("'")
        if not os.path.exists(path):
            print(f"Error: File '{path}' not found!")
            return get_file_or_folder(label)
        if not path.endswith(('.xlsx', '.xls')):
            print("Error: Please select an Excel file (.xlsx or .xls)")
            return get_file_or_folder(label)
        return {"type": "excel", "path": path}

    elif choice == "2":
        path = input(f"\nDrag and drop your {label} CSV file here (or type the path): ").strip().strip('"').strip("'")
        if not os.path.exists(path):
            print(f"Error: File '{path}' not found!")
            return get_file_or_folder(label)
        if not path.endswith('.csv'):
            print("Error: Please select a CSV file")
            return get_file_or_folder(label)
        return {"type": "csv", "path": path}

    elif choice == "3":
        path = input(f"\nDrag and drop the {label} folder here (or type the path): ").strip().strip('"').strip("'")
        if not os.path.isdir(path):
            print(f"Error: Folder '{path}' not found!")
            return get_file_or_folder(label)
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        if not csv_files:
            print(f"Error: No CSV files found in '{path}'!")
            return get_file_or_folder(label)
        print(f"Found {len(csv_files)} CSV files in the folder")
        return {"type": "csv_folder", "path": path}

    else:
        print("Invalid choice! Please try again.")
        return get_file_or_folder(label)


def select_column_from_file(file_info, label):
    """Select the matching column and optional additional columns from a file."""
    if file_info["type"] == "excel":
        xl_file = pd.ExcelFile(file_info["path"])
        sheets = xl_file.sheet_names

        if len(sheets) == 1:
            sheet = sheets[0]
            print(f"\nUsing sheet: {sheet}")
        else:
            print("\nAvailable sheets:")
            for i, s in enumerate(sheets, 1):
                print(f"  {i}. {s}")
            choice = int(input(f"Select sheet (1-{len(sheets)}): "))
            sheet = sheets[choice - 1]

        df = pd.read_excel(file_info["path"], sheet_name=sheet, nrows=5)
        file_info["sheet"] = sheet
        columns = df.columns.tolist()

    elif file_info["type"] == "csv":
        df = pd.read_csv(file_info["path"], nrows=5)
        columns = df.columns.tolist()

    elif file_info["type"] == "csv_folder":
        first_csv = glob.glob(os.path.join(file_info["path"], "*.csv"))[0]
        df = pd.read_csv(first_csv, nrows=5)
        columns = df.columns.tolist()

    print(f"\nPreview of {label} data (first 5 rows):")
    print(df.to_string(index=False))

    print(f"\nAvailable columns in {label}:")
    for i, col in enumerate(columns, 1):
        print(f"  {i}. {col}")

    choice = int(input(f"\nWhich column contains the values to match? (1-{len(columns)}): "))
    selected_column = columns[choice - 1]
    file_info["column"] = selected_column

    # Ask about additional columns (for single files only)
    remaining_cols = [col for col in columns if col != selected_column]
    if file_info["type"] in ["excel", "csv"] and remaining_cols:
        print(f"\nWould you like to include other columns from {label} in the output?")
        print("(These extra columns will be carried alongside each matched row.)")
        include_choice = input("Include ALL other columns? (y/n) [default: y]: ").strip().lower()

        if include_choice == 'n':
            additional = []
            print("\nSelect which columns to include:")
            for col in remaining_cols:
                if input(f"  Include '{col}'? (y/n): ").strip().lower() == 'y':
                    additional.append(col)
            file_info["additional_columns"] = additional
            if additional:
                print(f"  Including: {', '.join(additional)}")
            else:
                print("  No additional columns selected.")
        else:
            file_info["additional_columns"] = remaining_cols
            print(f"  Including all: {', '.join(remaining_cols)}")
    else:
        file_info["additional_columns"] = []

    return file_info


def get_output_config():
    """Get output file configuration."""
    print("\n=== STEP 3: Output Configuration ===")
    print("  1. Excel file (.xlsx) - includes metadata sheet")
    print("  2. CSV file (.csv)")

    output_choice = input("\nChoose output format (1-2) [default: 1]: ").strip() or "1"

    if output_choice == "1":
        output_path = input("Output file name [default: results.xlsx]: ").strip()
        if not output_path:
            output_path = "results.xlsx"
        if not output_path.endswith('.xlsx'):
            output_path += '.xlsx'
        return {"type": "excel", "path": output_path}
    else:
        output_path = input("Output file name [default: results.csv]: ").strip()
        if not output_path:
            output_path = "results.csv"
        if not output_path.endswith('.csv'):
            output_path += '.csv'
        return {"type": "csv", "path": output_path}


def get_advanced_settings():
    """Get optional advanced settings with sensible defaults."""
    print("\n=== STEP 4: Settings ===")
    use_advanced = input("Configure advanced settings? (y/n) [default: n]: ").strip().lower()

    if use_advanced == 'y':
        print("\nEmbedding model:")
        print("  1. text-embedding-3-large (best quality)")
        print("  2. text-embedding-3-small (faster/cheaper)")
        print("  3. text-embedding-ada-002 (legacy)")
        model_choice = input("Select model (1-3) [default: 1]: ").strip() or "1"
        model_map = {
            "1": "text-embedding-3-large",
            "2": "text-embedding-3-small",
            "3": "text-embedding-ada-002"
        }
        embedding_model = model_map.get(model_choice, "text-embedding-3-large")

        max_workers_input = input("Parallel threads (default: 10): ").strip()
        max_workers = int(max_workers_input) if max_workers_input else 10

        batch_size_input = input("Batch size (default: 100): ").strip()
        batch_size = int(batch_size_input) if batch_size_input else 100

        clear_cache = input("Clear embedding cache and regenerate? (y/n) [default: n]: ").strip().lower() == 'y'
    else:
        embedding_model = "text-embedding-3-large"
        max_workers = 10
        batch_size = 100
        clear_cache = False

    print(f"\n  Model: {embedding_model}")
    print(f"  Threads: {max_workers}, Batch size: {batch_size}")

    return embedding_model, max_workers, batch_size, clear_cache


# -------------------------------------------------------------------
# Cache / Backup Functions
# -------------------------------------------------------------------
def get_cache_filename(config):
    """Generate a cache filename based on the data source and column."""
    if config["type"] == "csv_folder":
        base = os.path.basename(config["path"].rstrip("/\\"))
    else:
        base = os.path.splitext(os.path.basename(config["path"]))[0]
    col = config["column"].replace(" ", "_").replace("/", "_")
    return f"cache_{base}_{col}_embeddings.pkl"


def load_pickle(file):
    """Load data from pickle file if it exists."""
    if os.path.exists(file):
        try:
            with open(file, "rb") as f:
                data = pickle.load(f)
            logging.info(f"Loaded cache from '{file}' ({len(data)} entries)")
            return data
        except Exception as e:
            logging.warning(f"Could not read cache file '{file}': {e}. Starting fresh.")
            return {}
    return {}


def save_pickle(data, file):
    """Save data to a pickle cache file."""
    with open(file, "wb") as f:
        pickle.dump(data, f)


# -------------------------------------------------------------------
# Data Loading Functions
# -------------------------------------------------------------------
def load_data(config):
    """Load data based on configuration. Returns (DataFrame, column_name)."""
    if config["type"] == "csv_folder":
        return deduplicate_csv_column(config["path"], config["column"])
    elif config["type"] == "excel":
        df = pd.read_excel(config["path"], sheet_name=config["sheet"])
        cols_to_load = [config["column"]] + config.get("additional_columns", [])
        return df[cols_to_load], config["column"]
    elif config["type"] == "csv":
        df = pd.read_csv(config["path"])
        cols_to_load = [config["column"]] + config.get("additional_columns", [])
        return df[cols_to_load], config["column"]


def deduplicate_csv_column(folder, column):
    """Load all CSVs from a folder, extract a column, and deduplicate."""
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    all_values = []

    for file in csv_files:
        try:
            df = pd.read_csv(file, usecols=[column], encoding="utf-8")
            all_values.append(df[column])
        except Exception as e:
            logging.warning(f"Error reading {file}: {e}")

    combined_series = pd.concat(all_values).drop_duplicates().reset_index(drop=True)
    return pd.DataFrame({column: combined_series}), column


# -------------------------------------------------------------------
# Text Processing Functions
# -------------------------------------------------------------------
def clean_text(text):
    """Standardize text for embedding."""
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    return text


# -------------------------------------------------------------------
# Embedding Functions with True Multithreading
# -------------------------------------------------------------------
class EmbeddingGenerator:
    def __init__(self, api_key, model, max_workers=10, max_retries=3):
        self.api_key = api_key
        self.model = model
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.client = OpenAI(api_key=api_key)

    def get_embedding(self, text):
        """Get embedding for a single text with retry logic."""
        text = str(text).replace("\n", " ")

        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=[text]
                )
                return text, response.data[0].embedding
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logging.error(f"Failed to get embedding for '{text[:50]}...' after {self.max_retries} attempts: {e}")
                    return text, None
                time.sleep(2 ** attempt)  # Exponential backoff

    def generate_embeddings_batch(self, texts, embeddings_cache, progress_bar):
        """Generate embeddings using ThreadPoolExecutor for true multithreading."""
        missing_texts = [text for text in texts if text not in embeddings_cache]

        if not missing_texts:
            progress_bar.update(len(texts))
            return embeddings_cache

        # Update progress for cached items
        progress_bar.update(len(texts) - len(missing_texts))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_text = {executor.submit(self.get_embedding, text): text
                             for text in missing_texts}

            for future in as_completed(future_to_text):
                text, embedding = future.result()
                if embedding:
                    embeddings_cache[text] = embedding
                progress_bar.update(1)

        return embeddings_cache


def generate_embeddings_multithreaded(text_list, pickle_file, model, api_key, max_workers=10, batch_size=100):
    """Generate embeddings with true multithreading, batching, and progress bar."""
    embeddings = load_pickle(pickle_file)

    cached_count = sum(1 for t in text_list if t in embeddings)
    need_count = len(text_list) - cached_count

    print(f"\n  Embedding Statistics:")
    print(f"    Total items: {len(text_list)}")
    print(f"    Already cached: {cached_count}")
    print(f"    Need to generate: {need_count}")
    print(f"    Cache file: {pickle_file}")

    missing_texts = [text for text in text_list if text not in embeddings]

    if missing_texts:
        generator = EmbeddingGenerator(api_key, model, max_workers)

        total_batches = (len(missing_texts) + batch_size - 1) // batch_size
        print(f"    Processing in {total_batches} batch(es) of up to {batch_size}")

        with tqdm(total=len(text_list), desc="Generating embeddings", unit="text") as pbar:
            # Update progress for already cached items
            pbar.update(len(text_list) - len(missing_texts))

            for batch_idx in range(0, len(missing_texts), batch_size):
                batch_texts = missing_texts[batch_idx:batch_idx + batch_size]
                batch_num = (batch_idx // batch_size) + 1

                pbar.set_description(f"Embeddings (Batch {batch_num}/{total_batches})")

                embeddings = generator.generate_embeddings_batch(batch_texts, embeddings, pbar)

                # Save after each batch for safety (backup checkpoint)
                save_pickle(embeddings, pickle_file)

                # Small delay between batches to avoid rate limits
                if batch_idx + batch_size < len(missing_texts):
                    time.sleep(0.5)
    else:
        print("    All embeddings already cached! (Using backup)")

    # Extract valid embeddings
    valid_embeddings = []
    valid_indices = []

    for i, text in enumerate(text_list):
        if text in embeddings and embeddings[text]:
            valid_embeddings.append(embeddings[text])
            valid_indices.append(i)

    if not valid_embeddings:
        raise ValueError("No valid embeddings found. Check your API key and data.")

    embedding_array = np.array(valid_embeddings, dtype=np.float32)

    return embedding_array, valid_indices


# -------------------------------------------------------------------
# FAISS Functions
# -------------------------------------------------------------------
def build_faiss_index(embeddings):
    """Creates a FAISS index for fast cosine similarity search."""
    d = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index


def faiss_match(query_emb, faiss_index, original_values):
    """Finds the closest match using FAISS. Returns (value, score, match_index)."""
    query_emb = np.array([query_emb]).astype(np.float32)
    faiss.normalize_L2(query_emb)

    distances, indices = faiss_index.search(query_emb, k=1)
    best_match_idx = indices[0][0]

    if best_match_idx < len(original_values):
        best_match = original_values[best_match_idx]
        similarity_score = distances[0][0]
        return best_match, similarity_score, best_match_idx
    else:
        logging.warning(f"Index {best_match_idx} is out of bounds")
        return "NO_MATCH", 0.0, -1


# -------------------------------------------------------------------
# Output Functions
# -------------------------------------------------------------------
def save_results(results_df, output_config):
    """Save results to file based on configuration."""
    if output_config["type"] == "excel":
        with pd.ExcelWriter(output_config["path"], engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='Fuzzy Matches', index=False)

            # Add metadata sheet
            metadata = pd.DataFrame({
                'Parameter': ['Generated Date', 'Total Matches', 'Average Similarity',
                             'Min Similarity', 'Max Similarity'],
                'Value': [
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    len(results_df),
                    f"{results_df['Similarity_Score'].mean():.3f}",
                    f"{results_df['Similarity_Score'].min():.3f}",
                    f"{results_df['Similarity_Score'].max():.3f}"
                ]
            })
            metadata.to_excel(writer, sheet_name='Metadata', index=False)

            # Format the Excel file - auto-adjust column widths
            worksheet = writer.sheets['Fuzzy Matches']
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width

    elif output_config["type"] == "csv":
        results_df.to_csv(output_config["path"], index=False)


# -------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------
def main():
    """Main function to run the fuzzy lookup tool."""
    setup_logging()

    print("\n=== Fuzzy Matching Tool ===")
    print("Match records between two files using AI-powered semantic similarity.")
    print("Uses OpenAI embeddings + FAISS for fast, accurate fuzzy matching.\n")

    # ------------------------------------------------------------------
    # STEP 0: API Key (asked upfront before anything else)
    # ------------------------------------------------------------------
    api_key = get_api_key()

    # ------------------------------------------------------------------
    # STEP 1: Select YOUR DATA file
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: Select YOUR DATA file")
    print("=" * 60)
    print("This is the file containing the records you want to find")
    print("matches for. Every row in this file will appear in the")
    print("output alongside its best match.")
    input_config = get_file_or_folder("YOUR DATA")
    input_config = select_column_from_file(input_config, "YOUR DATA")

    # ------------------------------------------------------------------
    # STEP 2: Select LOOKUP file
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Select your LOOKUP file")
    print("=" * 60)
    print("This is the reference list to match against. The tool will")
    print("search this file to find the best match for each row in")
    print("your data.")
    lookup_config = get_file_or_folder("LOOKUP")
    lookup_config = select_column_from_file(lookup_config, "LOOKUP")

    # ------------------------------------------------------------------
    # STEP 3: Output configuration
    # ------------------------------------------------------------------
    output_config = get_output_config()

    # ------------------------------------------------------------------
    # STEP 4: Advanced settings (optional, sensible defaults)
    # ------------------------------------------------------------------
    embedding_model, max_workers, batch_size, clear_cache = get_advanced_settings()

    # ------------------------------------------------------------------
    # Set up cache (pickle) file names based on input files
    # ------------------------------------------------------------------
    input_pickle = get_cache_filename(input_config)
    lookup_pickle = get_cache_filename(lookup_config)

    # Handle cache clearing if requested
    if clear_cache:
        for pf in [input_pickle, lookup_pickle]:
            if os.path.exists(pf):
                os.remove(pf)
                print(f"  Cleared cache: {pf}")

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Starting Processing")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    input_data, input_column = load_data(input_config)
    print(f"  YOUR DATA: {len(input_data)} records from '{input_config.get('path', 'folder')}'")

    lookup_data, lookup_column = load_data(lookup_config)
    print(f"  LOOKUP:    {len(lookup_data)} records from '{lookup_config.get('path', 'folder')}'")

    # Clean text
    print("\nCleaning text...")
    input_data_cleaned = input_data[input_column].apply(clean_text).tolist()
    lookup_data_cleaned = lookup_data[lookup_column].apply(clean_text).tolist()

    # Generate embeddings
    print("\nGenerating embeddings for YOUR DATA...")
    input_embs, input_valid_idx = generate_embeddings_multithreaded(
        input_data_cleaned, input_pickle, embedding_model, api_key, max_workers, batch_size
    )

    print("\nGenerating embeddings for LOOKUP data...")
    lookup_embs, lookup_valid_idx = generate_embeddings_multithreaded(
        lookup_data_cleaned, lookup_pickle, embedding_model, api_key, max_workers, batch_size
    )

    # Get valid original values for lookup
    lookup_vals_original = [lookup_data[lookup_column].iloc[i] for i in lookup_valid_idx]

    # Build FAISS index from lookup data
    print("\nBuilding search index from LOOKUP data...")
    lookup_faiss_index = build_faiss_index(lookup_embs)

    # Perform matching - find best lookup match for each row in your data
    print("\nPerforming fuzzy matching...")
    results = []

    input_additional = input_config.get("additional_columns", [])
    lookup_additional = lookup_config.get("additional_columns", [])

    def process_matching(i):
        """Match each input row to the best lookup entry."""
        if i < len(input_embs) and i < len(input_valid_idx):
            best_match, similarity, match_idx = faiss_match(
                input_embs[i], lookup_faiss_index, lookup_vals_original
            )
            original_idx = input_valid_idx[i]

            # Build result row in a sensible column order:
            # 1. Input matching column
            # 2. Additional columns from input (your data)
            # 3. Best match from lookup
            # 4. Additional columns from lookup
            # 5. Similarity score
            result = {}
            result[input_column] = input_data[input_column].iloc[original_idx]

            # Additional columns from YOUR DATA
            for col in input_additional:
                if col in input_data.columns:
                    result[col] = input_data[col].iloc[original_idx]

            result["Best_Match"] = best_match

            # Additional columns from LOOKUP (from the matched row)
            if match_idx >= 0 and lookup_additional:
                lookup_original_idx = lookup_valid_idx[match_idx]
                for col in lookup_additional:
                    if col in lookup_data.columns:
                        # Prefix with "Matched_" if the column name conflicts
                        output_col = f"Matched_{col}" if col in result else col
                        result[output_col] = lookup_data[col].iloc[lookup_original_idx]

            result["Similarity_Score"] = similarity
            return result
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_matching, i) for i in range(len(input_embs))]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Matching"):
            result = future.result()
            if result:
                results.append(result)

    # Create results DataFrame
    df_results = pd.DataFrame(results)

    # Save results
    print(f"\nSaving results to {output_config['path']}...")
    save_results(df_results, output_config)

    # Summary statistics
    if not df_results.empty:
        print(f"\n=== Results Summary ===")
        print(f"  Total matches: {len(df_results)}")
        print(f"  Average similarity: {df_results['Similarity_Score'].mean():.3f}")
        print(f"  Min similarity: {df_results['Similarity_Score'].min():.3f}")
        print(f"  Max similarity: {df_results['Similarity_Score'].max():.3f}")

        # Show sample matches
        print(f"\n=== Sample Matches (Top 10) ===")
        print(df_results.nlargest(10, 'Similarity_Score').to_string(index=False))

        # Ask if user wants to filter by similarity threshold
        print("\nWould you like to filter results by a minimum similarity threshold?")
        if input("Apply threshold filter? (y/n) [default: n]: ").strip().lower() == 'y':
            threshold = float(input("Enter minimum similarity threshold (0-1): "))
            filtered_df = df_results[df_results['Similarity_Score'] >= threshold]

            if output_config["type"] == "excel":
                with pd.ExcelWriter(output_config["path"], mode='a', engine='openpyxl') as writer:
                    filtered_df.to_excel(writer, sheet_name=f'Filtered (>={threshold})', index=False)
                print(f"\n  Filtered results ({len(filtered_df)} matches) added as new sheet in Excel file.")
            else:
                filtered_path = output_config["path"].replace('.csv', f'_filtered_{threshold}.csv')
                filtered_df.to_csv(filtered_path, index=False)
                print(f"\n  Filtered results saved to '{filtered_path}'.")

    print("\n=== Process Complete! ===")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise
