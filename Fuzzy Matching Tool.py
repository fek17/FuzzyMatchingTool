import pandas as pd
import numpy as np
import os
import re
import logging
import pickle
import faiss
import glob
import asyncio
import aiohttp
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from pathlib import Path
from datetime import datetime
import threading
import queue
import time

# -------------------------------------------------------------------
# Configuration & Logging -t est
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
    """Get OpenAI API key from user or environment"""
    api_key = "xx"
    if not api_key:
        print("\n=== OpenAI API Configuration ===")
        print("No OPENAI_API_KEY found in environment variables.")
        api_key = input("Please enter your OpenAI API key: ").strip()
    return api_key

def get_file_or_folder():
    """More intuitive file/folder selection"""
    print("\n=== Select Your Data Source ===")
    print("What would you like to match?")
    print("1. A single Excel file")
    print("2. A single CSV file") 
    print("3. Multiple CSV files in a folder")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        path = input("\nDrag and drop your Excel file here (or type the path): ").strip().strip('"')
        if not os.path.exists(path):
            print(f"Error: File '{path}' not found!")
            return get_file_or_folder()
        if not path.endswith(('.xlsx', '.xls')):
            print("Error: Please select an Excel file (.xlsx or .xls)")
            return get_file_or_folder()
        return {"type": "excel", "path": path}
    
    elif choice == "2":
        path = input("\nDrag and drop your CSV file here (or type the path): ").strip().strip('"')
        if not os.path.exists(path):
            print(f"Error: File '{path}' not found!")
            return get_file_or_folder()
        if not path.endswith('.csv'):
            print("Error: Please select a CSV file")
            return get_file_or_folder()
        return {"type": "csv", "path": path}
    
    elif choice == "3":
        path = input("\nDrag and drop the folder here (or type the path): ").strip().strip('"')
        if not os.path.isdir(path):
            print(f"Error: Folder '{path}' not found!")
            return get_file_or_folder()
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        if not csv_files:
            print(f"Error: No CSV files found in '{path}'!")
            return get_file_or_folder()
        print(f"Found {len(csv_files)} CSV files in the folder")
        return {"type": "csv_folder", "path": path}
    
    else:
        print("Invalid choice! Please try again.")
        return get_file_or_folder()

def select_column_from_file(file_info):
    """Select column from file with preview"""
    if file_info["type"] == "excel":
        xl_file = pd.ExcelFile(file_info["path"])
        sheets = xl_file.sheet_names
        
        if len(sheets) == 1:
            sheet = sheets[0]
            print(f"\nUsing sheet: {sheet}")
        else:
            print("\nAvailable sheets:")
            for i, s in enumerate(sheets, 1):
                print(f"{i}. {s}")
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
    
    print("\nColumn preview:")
    print(df.to_string(index=False))
    
    print("\nAvailable columns:")
    for i, col in enumerate(columns, 1):
        print(f"{i}. {col}")
    
    choice = int(input(f"\nWhich column contains the values to match? (1-{len(columns)}): "))
    selected_column = columns[choice - 1]
    file_info["column"] = selected_column
    
    # Ask about additional columns for single files
    if file_info["type"] in ["excel", "csv"]:
        print("\nWould you like to include other columns in the output?")
        remaining_cols = [col for col in columns if col != selected_column]
        
        if remaining_cols:
            include_all = input("Include ALL other columns? (y/n): ").strip().lower() == 'y'
            
            if include_all:
                file_info["additional_columns"] = remaining_cols
            else:
                additional = []
                print("\nSelect additional columns to include:")
                for col in remaining_cols:
                    if input(f"Include '{col}'? (y/n): ").strip().lower() == 'y':
                        additional.append(col)
                file_info["additional_columns"] = additional
        else:
            file_info["additional_columns"] = []
    
    return file_info

def get_configurations():
    """Simplified configuration gathering"""
    print("\n=== Fuzzy Lookup Configuration ===")
    
    # Reference file (what to match against)
    print("\nSTEP 1: Select your REFERENCE file (the file containing values to match AGAINST)")
    reference_config = get_file_or_folder()
    reference_config = select_column_from_file(reference_config)
    
    # Source file (what to match)
    print("\n\nSTEP 2: Select your SOURCE file (the file containing values you want to MATCH)")
    source_config = get_file_or_folder()
    source_config = select_column_from_file(source_config)
    
    # Output configuration
    print("\n\nSTEP 3: Output Configuration")
    print("1. Save as Excel file (.xlsx)")
    print("2. Save as CSV file (.csv)")
    
    output_choice = input("\nChoose output format (1-2) [default: 1]: ").strip() or "1"
    
    if output_choice == "1":
        output_path = input("Output file name (e.g., results.xlsx): ").strip()
        if not output_path.endswith('.xlsx'):
            output_path += '.xlsx'
        output_config = {"type": "excel", "path": output_path}
    else:
        output_path = input("Output file name (e.g., results.csv): ").strip()
        if not output_path.endswith('.csv'):
            output_path += '.csv'
        output_config = {"type": "csv", "path": output_path}
    
    return source_config, reference_config, output_config

# -------------------------------------------------------------------
# Data Loading Functions
# -------------------------------------------------------------------
def load_data(config):
    """Load data based on configuration"""
    if config["type"] == "csv_folder":
        return deduplicate_csv_column(config["path"], config["column"])
    elif config["type"] == "excel":
        df = pd.read_excel(config["path"], sheet_name=config["sheet"])
        if "additional_columns" in config:
            return df[[config["column"]] + config["additional_columns"]], config["column"]
        return df[[config["column"]]], config["column"]
    elif config["type"] == "csv":
        df = pd.read_csv(config["path"])
        if "additional_columns" in config:
            return df[[config["column"]] + config["additional_columns"]], config["column"]
        return df[[config["column"]]], config["column"]

def deduplicate_csv_column(folder, column):
    """Load all CSVs from a folder, extract a column, and deduplicate"""
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
    """Standardize text for embedding"""
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
        """Get embedding for a single text with retry logic"""
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
        """Generate embeddings using ThreadPoolExecutor for true multithreading"""
        missing_texts = [text for text in texts if text not in embeddings_cache]
        
        if not missing_texts:
            progress_bar.update(len(texts))
            return embeddings_cache
        
        # Update progress for cached items
        progress_bar.update(len(texts) - len(missing_texts))
        
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_text = {executor.submit(self.get_embedding, text): text 
                             for text in missing_texts}
            
            # Process completed tasks
            for future in as_completed(future_to_text):
                text, embedding = future.result()
                if embedding:
                    results[text] = embedding
                    embeddings_cache[text] = embedding
                progress_bar.update(1)
        
        return embeddings_cache

def load_pickle(file):
    """Load data from pickle file if it exists"""
    if os.path.exists(file):
        with open(file, "rb") as f:
            return pickle.load(f)
    return {}

def save_pickle(data, file):
    """Save data to a pickle file"""
    with open(file, "wb") as f:
        pickle.dump(data, f)

def generate_embeddings_multithreaded(text_list, pickle_file, model, api_key, max_workers=10, batch_size=100):
    """Generate embeddings with true multithreading, batching, and progress bar"""
    embeddings = load_pickle(pickle_file)
    
    print(f"\n📊 Embedding Statistics:")
    print(f"  - Total items: {len(text_list)}")
    print(f"  - Already cached: {sum(1 for t in text_list if t in embeddings)}")
    print(f"  - Need to generate: {sum(1 for t in text_list if t not in embeddings)}")
    
    missing_texts = [text for text in text_list if text not in embeddings]
    
    if missing_texts:
        generator = EmbeddingGenerator(api_key, model, max_workers)
        
        # Process in batches
        total_batches = (len(missing_texts) + batch_size - 1) // batch_size
        print(f"  - Processing in {total_batches} batches of {batch_size}")
        
        with tqdm(total=len(text_list), desc="Generating embeddings", unit="text") as pbar:
            # Update progress for already cached items
            pbar.update(len(text_list) - len(missing_texts))
            
            # Process each batch
            for batch_idx in range(0, len(missing_texts), batch_size):
                batch_texts = missing_texts[batch_idx:batch_idx + batch_size]
                batch_num = (batch_idx // batch_size) + 1
                
                pbar.set_description(f"Embeddings (Batch {batch_num}/{total_batches})")
                
                # Generate embeddings for this batch
                embeddings = generator.generate_embeddings_batch(batch_texts, embeddings, pbar)
                
                # Save after each batch (for safety)
                save_pickle(embeddings, pickle_file)
                
                # Small delay between batches to avoid rate limits
                if batch_idx + batch_size < len(missing_texts):
                    time.sleep(0.5)
    else:
        print("  - All embeddings already cached!")
    
    # Extract valid embeddings
    valid_embeddings = []
    valid_indices = []
    
    for i, text in enumerate(text_list):
        if text in embeddings and embeddings[text]:
            valid_embeddings.append(embeddings[text])
            valid_indices.append(i)
    
    if not valid_embeddings:
        raise ValueError("No valid embeddings found")
    
    embedding_array = np.array(valid_embeddings, dtype=np.float32)
    
    return embedding_array, valid_indices

# -------------------------------------------------------------------
# FAISS Functions
# -------------------------------------------------------------------
def build_faiss_index(embeddings):
    """Creates a FAISS index for fast cosine similarity search"""
    d = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(d)  
    index.add(embeddings)
    return index

def faiss_match(query_emb, faiss_index, original_values):
    """Finds the closest match using FAISS"""
    query_emb = np.array([query_emb]).astype(np.float32)
    faiss.normalize_L2(query_emb)

    distances, indices = faiss_index.search(query_emb, k=1)
    best_match_idx = indices[0][0]
    
    if best_match_idx < len(original_values):
        best_match = original_values[best_match_idx]
        similarity_score = distances[0][0]
        return best_match, similarity_score
    else:
        logging.warning(f"Index {best_match_idx} is out of bounds")
        return "NO_MATCH", 0.0

# -------------------------------------------------------------------
# Output Functions
# -------------------------------------------------------------------
def save_results(results_df, output_config):
    """Save results to file based on configuration"""
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
            
            # Format the Excel file
            workbook = writer.book
            worksheet = writer.sheets['Fuzzy Matches']
            
            # Auto-adjust column widths
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
    """Main function to run the fuzzy lookup tool"""
    setup_logging()
    
    print("\n🔍 === Generic Fuzzy Lookup Tool ===")
    print("This tool uses OpenAI embeddings and FAISS for semantic matching\n")
    
    # Get API configuration
    api_key = get_api_key()
    
    # Get embedding model
    print("\n=== Embedding Model Selection ===")
    print("1. text-embedding-3-large (best quality)")
    print("2. text-embedding-3-small (faster/cheaper)")
    print("3. text-embedding-ada-002 (legacy)")
    
    model_choice = input("\nSelect model (1-3) [default: 1]: ").strip() or "1"
    model_map = {
        "1": "text-embedding-3-large",
        "2": "text-embedding-3-small",
        "3": "text-embedding-ada-002"
    }
    embedding_model = model_map.get(model_choice, "text-embedding-3-large")
    
    # Performance settings
    print("\n=== Performance Settings ===")
    max_workers = input("Number of parallel threads for embeddings (default: 10): ").strip()
    max_workers = int(max_workers) if max_workers else 10
    1
    batch_size = input("Batch size for processing (default: 100): ").strip()
    batch_size = int(batch_size) if batch_size else 100
    
    # Get configurations
    source_config, reference_config, output_config = get_configurations()
    
    # Set up pickle file names
    source_pickle = "source_elec_embeddings_new.pkl"
    reference_pickle = "reference_elec_embeddings_new.pkl"
    
    print("\n\n🚀 === Starting Processing ===")
    
    # Load data
    print("\n📁 Loading data...")
    source_data, source_column = load_data(source_config)
    print(f"✓ Loaded {len(source_data)} records from source")
    
    reference_data, reference_column = load_data(reference_config)
    print(f"✓ Loaded {len(reference_data)} records from reference")
    
    # Clean text
    print("\n🧹 Cleaning text...")
    source_data_cleaned = source_data[source_column].apply(clean_text).tolist()
    reference_data_cleaned = reference_data[reference_column].apply(clean_text).tolist()
    
    # Generate embeddings
    print("\n🧠 Generating embeddings for source data...")
    source_embs, source_valid_idx = generate_embeddings_multithreaded(
        source_data_cleaned, source_pickle, embedding_model, api_key, max_workers, batch_size
    )
    
    print("\n🧠 Generating embeddings for reference data...")
    reference_embs, reference_valid_idx = generate_embeddings_multithreaded(
        reference_data_cleaned, reference_pickle, embedding_model, api_key, max_workers, batch_size
    )
    
    # Get valid original values
    source_vals_original = [source_data[source_column].iloc[i] for i in source_valid_idx]
    
    # Build FAISS index
    print("\n🏗️ Building search index...")
    source_faiss_index = build_faiss_index(source_embs)
    
    # Process matching
    print("\n🔗 Performing fuzzy matching...")
    results = []
    
    def process_matching(i):
        """Match reference items to source"""
        if i < len(reference_embs) and i < len(reference_valid_idx):
            best_match, similarity = faiss_match(reference_embs[i], source_faiss_index, source_vals_original)
            original_idx = reference_valid_idx[i]
            
            result = {
                "Reference_Value": reference_data[reference_column].iloc[original_idx], 
                "Best_Match": best_match, 
                "Similarity_Score": similarity
            }
            
            # Add additional columns if they exist
            if "additional_columns" in reference_config:
                for col in reference_config["additional_columns"]:
                    if col in reference_data.columns:
                        result[col] = reference_data[col].iloc[original_idx]
            
            return result
        return None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_matching, i) for i in range(len(reference_embs))]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Matching"):
            result = future.result()
            if result:
                results.append(result)
    
    # Create results DataFrame
    df_results = pd.DataFrame(results)
    
    # Save results
    print(f"\n💾 Saving results to {output_config['path']}...")
    save_results(df_results, output_config)
    
    # Summary statistics
    if not df_results.empty:
        print(f"\n✅ === Results Summary ===")
        print(f"📊 Total matches: {len(df_results)}")
        print(f"📈 Average similarity: {df_results['Similarity_Score'].mean():.3f}")
        print(f"📉 Min similarity: {df_results['Similarity_Score'].min():.3f}")
        print(f"📈 Max similarity: {df_results['Similarity_Score'].max():.3f}")
        
        # Show some example matches
        print("\n🔍 === Sample Matches (Top 10) ===")
        print(df_results.nlargest(10, 'Similarity_Score').to_string(index=False))
        
        # Ask if user wants to filter by similarity threshold
        print("\n\n🎯 Would you like to filter results by similarity threshold?")
        if input("Apply threshold filter? (y/n) [default: n]: ").strip().lower() == 'y':
            threshold = float(input("Enter minimum similarity threshold (0-1): "))
            filtered_df = df_results[df_results['Similarity_Score'] >= threshold]
            
            if output_config["type"] == "excel":
                # Save filtered results to a new sheet
                with pd.ExcelWriter(output_config["path"], mode='a', engine='openpyxl') as writer:
                    filtered_df.to_excel(writer, sheet_name=f'Filtered (>={threshold})', index=False)
                print(f"\n✓ Filtered results ({len(filtered_df)} matches) added to Excel file.")
            else:
                # Save filtered results to a new CSV
                filtered_path = output_config["path"].replace('.csv', f'_filtered_{threshold}.csv')
                filtered_df.to_csv(filtered_path, index=False)
                print(f"\n✓ Filtered results saved to '{filtered_path}'.")
    
    print("\n🎉 === Process Complete! ===")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Operation cancelled by user.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise